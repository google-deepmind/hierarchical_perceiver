# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Blocks for the Perceiver IO and HiP."""

from typing import Dict, List, Mapping, Optional, Tuple

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
from jax import numpy as jnp
import perceiver_helpers

BATCH_DIM = 0
GROUPS_DIM = 1
INDEX_DIM = -2
CHANNELS_DIM = -1

RECONSTRUCTION_HEAD_NAME = 'reconstruction_head'


def regroup(
    inputs: chex.Array,
    num_output_groups: int,
    regroup_type: str,
) -> chex.Array:
  """Re-group an input array from [B, G, N, C] to [B, G', N', C].

  Args:
    inputs: the array to regroup.
    num_output_groups: The number of output groups G'.
    regroup_type: The regrouping strategy to use.
  Returns:
    The re-grouped array.
  """
  batch_size = inputs.shape[BATCH_DIM]
  num_input_groups = inputs.shape[GROUPS_DIM]
  num_input_latents = inputs.shape[INDEX_DIM]
  num_channels = inputs.shape[CHANNELS_DIM]

  if regroup_type in ['reshape', 'transpose_reshape']:
    new_index_dim = num_input_groups * num_input_latents // num_output_groups
    if regroup_type == 'transpose_reshape':
      # [B, G, N, C] -> [B, N, G, C]
      # This leads to mixing between all input groups, rather than preferential
      # mixing between neighboring groups.
      inputs = jnp.swapaxes(inputs, 1, 2)
    outputs = jnp.reshape(
        inputs, (batch_size, num_output_groups, new_index_dim, num_channels))
  else:
    raise ValueError(f'Unknown regroup_type: {regroup_type}.')

  return outputs


class HiPCrossAttention(hk.Module):
  """A HiPCrossAttention module, including a dense block.

  Maps batched, grouped arrays of shape B x G x M x C to arrays of shape
  B x G x N x D.
  """

  def __init__(self,
               output_index_dim_train: Optional[int] = None,
               output_index_dim_eval: Optional[int] = None,
               output_num_channels: Optional[int] = None,
               activation_name: str = 'sq_relu',
               widening_factor: int = 1,
               num_heads: int = 8,
               use_post_attention_residual: bool = False,
               name: Optional[str] = None):
    """Constructs a new HiPCrossAttention.

    Args:
      output_index_dim_train: The output index dimension size at train. Ignored
        if `query_inputs` is specified directly at call.
      output_index_dim_eval: The output index dimension size at eval. Ignored
        if `query_inputs` is specified directly at call.
      output_num_channels: The number of output channels.
      activation_name: The activation to use.
      widening_factor: The widening factor to use in the output MLP.
      num_heads: The number of heads to use in cross-attention.
      use_post_attention_residual: Enable the post-attention residual
        connection? This residual adds the query inputs to the output of
        attention.
      name: Haiku module name.
    """
    super(HiPCrossAttention, self).__init__(name=name)
    self._output_index_dim_train = output_index_dim_train
    self._output_index_dim_eval = output_index_dim_eval

    self._output_num_channels = output_num_channels
    self._activation_name = activation_name

    self._widening_factor = widening_factor
    self._num_heads = num_heads
    self._use_post_attention_residual = use_post_attention_residual

  def _subsample_query_inputs(self, query_inputs):
    """Randomly subsample the number of query inputs.

    Args:
      query_inputs: An array of latent query inputs, of shape
        B x G x N_eval x C

    Returns:
      A subsampled array of latent query inputs, of shape
        B x G x N_train x C,
        where N_train < N_eval.
    """
    batch_size, num_groups, _, _ = query_inputs.shape
    def get_weight_indices():
      # Sample indices without replacement for each batch & group.
      rng_keys = hk.next_rng_keys(batch_size * num_groups)
      rng_keys = jnp.reshape(
          jnp.asarray(rng_keys), [batch_size, num_groups, -1])
      def get_per_group_weights(random_key):
        # Get the weight indices for a single group in a single batch.
        weight_indices = jnp.arange(0, self._output_index_dim_eval)
        weight_indices = jax.random.shuffle(random_key, weight_indices)
        return weight_indices
      # Subsample outside of the vmap to avoid compiling to a while loop.
      weight_indices = jax.vmap(jax.vmap(get_per_group_weights))(rng_keys)
      # [B, G, train_index_dim]
      return weight_indices[..., :self._output_index_dim_train]

    weight_indices = get_weight_indices()
    # One-hot index onto the full weights (note: uses a bfloat16 matmul).
    # [B, G, train_index_dim] -> [B, G, train_index_dim, eval_index_dim]
    one_hot_indices = jax.nn.one_hot(
        weight_indices, num_classes=self._output_index_dim_eval)
    # [B, G, train_index_dim, C]
    query_inputs = jnp.einsum(
        'bgMc,bgmM->bgmc', query_inputs, one_hot_indices)
    return query_inputs

  def __call__(
      self,
      inputs: chex.Array,
      *,
      query_inputs: Optional[chex.Array] = None,
      pre_attention_residual: Optional[chex.Array] = None,
      attention_mask: Optional[chex.Array] = None,
      is_training: bool = True,
  ) -> chex.Array:
    """Calls the HiPCrossAttention.

    Args:
      inputs: An input array of shape B x G x M x C to cross-attend to.
      query_inputs: Optional query inputs to the cross-attention. If provided,
        learned latent queries will not be constructed. Typically used for
        decoding from position encoding queries.
      pre_attention_residual: An optional array that will be added to the
        queries before the cross-attention. Used for U-Net-like skip connections
        in HiP.
      attention_mask: An optional mask for cross-attention.
      is_training: Are we currently training, yes or no?
    Returns:
      The array after processing with the HiPCrossAttention.
    """
    # Input shape is assumed to be
    # [batch_size, num_groups, index_dim_per_group, num_channels]
    batch_size, num_groups, _, _ = inputs.shape

    # If explicit query_inputs are not provided, learn latent queries.
    if query_inputs is None:
      assert self._output_index_dim_train is not None
      assert self._output_index_dim_eval is not None
      assert self._output_index_dim_eval >= self._output_index_dim_train

      assert self._output_num_channels is not None

      query_inputs = perceiver_helpers.TrainablePositionEncoding(
          # The underlying array contains all latents expected at eval time
          index_dim=num_groups * self._output_index_dim_eval,
          num_channels=self._output_num_channels,
          name='query_inputs')(batch_size=batch_size)
      # Fold groups into the batch dimension
      query_inputs = einshape('b(gm)c->bgmc', query_inputs, g=num_groups)

      if is_training and (
          self._output_index_dim_train < self._output_index_dim_eval):
        # Sample a random subset of latent queries for this training batch.
        query_inputs = self._subsample_query_inputs(query_inputs)

    output_index_dim = query_inputs.shape[-2]
    output_num_channels = query_inputs.shape[-1]

    if pre_attention_residual is not None:
      assert pre_attention_residual.shape[-2] == output_index_dim
      # Project pre_attention_residual to the correct shape.
      residual_num_channels = pre_attention_residual.shape[-1]
      if residual_num_channels != output_num_channels:
        pre_attention_residual = perceiver_helpers.conv_1d(
            output_channels=output_num_channels,
            name='pre_attention_residual_linear')(pre_attention_residual)

      query_inputs += pre_attention_residual

    # -----------------------------------------
    # ---------- Cross-attend -> MLP ----------
    # -----------------------------------------
    attention = perceiver_helpers.Attention(
        num_heads=self._num_heads,
        # KV input channels determine the dimension of the attention matmul.
        qk_channels=inputs.shape[-1],
        v_channels=inputs.shape[-1],
        # (Latent) query channels determine the size of the output.
        output_channels=query_inputs.shape[-1])(
            inputs_q=perceiver_helpers.layer_norm(query_inputs),
            inputs_kv=perceiver_helpers.layer_norm(inputs),
            attention_mask=attention_mask)

    if self._use_post_attention_residual:
      attention += query_inputs

    output = attention
    output += perceiver_helpers.Dense(
        widening_factor=self._widening_factor)(
            perceiver_helpers.layer_norm(attention), is_training=is_training)

    return output


class SelfAttention(hk.Module):
  """A self-attention module, including a dense block."""

  def __init__(self,
               widening_factor: int = 4,
               dropout_prob: float = 0.0,
               dropout_attn_prob: float = 0.0,
               drop_path_rate: float = 0.0,
               num_heads: int = 8,
               att_init_scale: float = 1.0,
               dense_init_scale: float = 1.0,
               qk_channels: Optional[int] = None,
               v_channels: Optional[int] = None,
               activation_name: str = 'sq_relu',
               name: Optional[str] = None):
    super(SelfAttention, self).__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._dropout_attn_prob = dropout_attn_prob
    self._num_heads = num_heads
    self._att_init_scale = att_init_scale
    self._dense_init_scale = dense_init_scale
    self._qk_channels = qk_channels
    self._v_channels = v_channels
    self._activation_name = activation_name

    if drop_path_rate > 0.:
      self._drop_path = perceiver_helpers.StochasticDepth(drop_path_rate)
    else:
      self._drop_path = lambda x, _: x

  def __call__(self,
               inputs: chex.Array,
               attention_mask: Optional[chex.Array] = None,
               is_training: bool = True) -> chex.Array:
    dropout_prob = self._dropout_prob if is_training else 0.0
    x = inputs

    dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0
    qkv_inputs = perceiver_helpers.layer_norm(inputs)
    attention = perceiver_helpers.Attention(
        num_heads=self._num_heads,
        init_scale=self._att_init_scale,
        qk_channels=self._qk_channels,
        v_channels=self._v_channels,
        dropout_prob=dropout_attn_prob)(qkv_inputs, qkv_inputs,
                                        attention_mask=attention_mask)
    attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)

    x = x + self._drop_path(attention, is_training)

    dense_layer = perceiver_helpers.Dense(
        widening_factor=self._widening_factor,
        dropout_prob=dropout_prob,
        init_scale=self._dense_init_scale)
    dense_out = dense_layer(perceiver_helpers.layer_norm(x),
                            is_training=is_training)
    x += self._drop_path(dense_out, is_training)
    return x


class PerceiverBlock(hk.Module):
  """The PerceiverBlock combines regrouping, cross- and self-attention."""

  def __init__(self,
               *,
               num_output_groups: int,
               output_index_dim: int,
               num_output_channels: int,
               num_self_attend_layers: int,
               num_self_attend_heads: int,
               self_attend_widening_factor: int = 4,
               num_cross_attend_heads: int = 1,
               cross_attend_widening_factor: int = 1,
               regroup_inputs: bool = True,
               regroup_type: str = 'reshape',
               activation_name: str = 'sq_relu',
               use_post_attention_residual: bool = True,
               output_index_dim_train: Optional[int] = None,
               output_index_dim_eval: Optional[int] = None,
               dropout_prob: float = 0.0,
               drop_path_rate: float = 0.0,
               name: str):
    """Constructs a new PerceiverBlock.

    The constructed block will reshape inputs into num_output_groups, then pass
    them through a cross-attention (HiPCrossAttention), followed by a series of
    self-attentions (n = num_self_attend_layers). num_output_channels and
    output_index_dim control the output size.

    The input must be (batch, groups, index_dim, channels), whereas
    the groups*index must be divisible by num_output_groups. The output size
    will be (batch_size, output_groups, output_index_dim, num_output_channels),
    where only batch_size must remain constant with the input.

    Args:
      num_output_groups: Number of groups for this block.
      output_index_dim: The default index dimension size at train time.
      num_output_channels: The number of output channels.
      num_self_attend_layers: Number of self-attend layers.
      num_self_attend_heads: Number of heads per self-attend layer.
      self_attend_widening_factor: SelfAttention widening factor.
      num_cross_attend_heads: Number of HiPCrossAttention heads.
      cross_attend_widening_factor: HiPCrossAttention widening factor.
      regroup_inputs: If True, the input array will be restructured to match the
        number of output groups. If False, the input will be passed in as is. If
        False, the number of input and output groups must match.
      regroup_type: The regrouping strategy to use.
      activation_name: Activation for HiPCrossAttention and SelfAttention.
      use_post_attention_residual: Enable the post-attention residual
        connection? This residual adds the query inputs to the output of
        attention.
      output_index_dim_train: The optional output index dimension size at train.
        If specified, overrides output_index_dim at train time.
      output_index_dim_eval: The optional output index dimension size at eval.
        If specified, overrides output_index_dim at eval time.
      dropout_prob: SelfAttention dropout probability.
      drop_path_rate: SelfAttention drop path rate.
      name: Haiku module name.
    """

    super().__init__(name=name)

    self.num_output_groups = num_output_groups
    self.num_output_channels = num_output_channels
    self.regroup_inputs = regroup_inputs
    self.regroup_type = regroup_type

    # Optionally specialize index dim for more compute at test time.
    # Usage: override `output_index_dim_train` to subsample at train and use the
    # default index dim at eval, override `output_index_dim_eval` to supersample
    # at eval and use the default index dim at train.
    if output_index_dim_train is not None and output_index_dim_eval is not None:
      raise ValueError(
          'Only one of `output_index_dim_train` and `output_index_dim_eval`'
          'should be overridden.')

    if output_index_dim_train is None:
      assert output_index_dim is not None, (
          'output_index_dim must be specified '
          'if output_index_dim_train is None.')
      self.output_index_dim_train = output_index_dim
    else:
      self.output_index_dim_train = output_index_dim_train

    if output_index_dim_eval is None:
      assert output_index_dim is not None, (
          'output_index_dim must be specified '
          'if output_index_dim_eval is None.')
      self.output_index_dim_eval = output_index_dim
    else:
      self.output_index_dim_eval = output_index_dim_eval

    assert self.output_index_dim_eval >= self.output_index_dim_train, (
        f'output_index_dim_eval (got {self.output_index_dim_eval}) must be '
        f'at least as big as output_index_dim_train '
        '(got {self.output_index_dim_train}).')

    assert (
        num_output_channels % num_self_attend_heads == 0
    ), f'num_self_attend_heads ({num_self_attend_heads}) should divide num_output_channels ({num_output_channels}) evenly'
    assert (
        num_output_channels % num_cross_attend_heads == 0
    ), f'num_cross_attend_heads ({num_cross_attend_heads}) should divide num_output_channels ({num_output_channels}) evenly'

    self.projector = HiPCrossAttention(
        activation_name=activation_name,
        num_heads=num_cross_attend_heads,
        output_num_channels=num_output_channels,
        use_post_attention_residual=use_post_attention_residual,
        widening_factor=cross_attend_widening_factor,
        output_index_dim_train=self.output_index_dim_train,
        output_index_dim_eval=self.output_index_dim_eval,
    )

    self.self_attentions = []
    for idx in range(num_self_attend_layers):
      self.self_attentions.append(
          SelfAttention(
              activation_name=activation_name,
              dropout_prob=dropout_prob,
              drop_path_rate=drop_path_rate,
              num_heads=num_self_attend_heads,
              widening_factor=self_attend_widening_factor,
              name=f'self_attend_id_{idx}'))

  def __call__(self, inputs: chex.ArrayTree, *,
               is_training: bool,
               pre_attention_residual: Optional[chex.ArrayTree] = None,
               attention_mask: Optional[chex.Array] = None) -> chex.ArrayTree:
    assert len(inputs.shape) == 4  # (batch, groups, index, channels)

    if is_training:
      output_index_dim = self.output_index_dim_train
    else:
      output_index_dim = self.output_index_dim_eval

    # Optionally regroup (this often means "reshape") inputs to a different
    # number of output groups. Elements in different groups can't interact.
    # Typically used in HiP.
    if self.regroup_inputs:
      inputs = regroup(
          inputs=inputs,
          num_output_groups=self.num_output_groups,
          regroup_type=self.regroup_type)
    else:
      chex.assert_equal(inputs.shape[GROUPS_DIM], self.num_output_groups)

    z = self.projector(
        inputs=inputs,
        pre_attention_residual=pre_attention_residual,
        is_training=is_training,
        attention_mask=attention_mask)
    chex.assert_shape(z, (inputs.shape[BATCH_DIM], self.num_output_groups,
                          output_index_dim, self.num_output_channels))

    for self_attend in self.self_attentions:
      z = self_attend(z, is_training=is_training)
      chex.assert_shape(z, (inputs.shape[BATCH_DIM], self.num_output_groups,
                            output_index_dim, self.num_output_channels))

    return z


class Embedder(hk.Module):
  """Projects inputs to the target number of channels.

  Inputs should be a dictionary of {modality_name:
  (batch_size, index_dim, num_channels)}. The output format will be similar, but
  with the new number of channels.

  Note both inputs and outputs are ungrouped. Grouping is handled by the
  Grouper module.
  """

  def __init__(self,
               *,
               num_embedding_channels: int,
               with_bias: bool = True,
               name: str = 'embedder'):
    super().__init__(name=name)
    self.with_bias = with_bias
    self.num_embedding_channels = num_embedding_channels
    self._orig_channels = None

  def embed(self, inputs: Mapping[str, chex.Array]) -> Dict[str, chex.Array]:
    """Takes raw inputs and embeds them to num_embedding_channels.

    Args:
      inputs: A dictionary of modality name and (batch, index, channels) value.

    Returns:
      A dictionary of modality name and (batch, index, num_embedding_channels).
    """
    _assert_input_shapes(inputs, expected_rank=3)

    self._orig_channels = {}
    out = {}
    for modality_name, value in inputs.items():
      # value: (batch, index, channels)

      self._orig_channels[modality_name] = value.shape[CHANNELS_DIM]

      conv_1d_embed = hk.Linear(
          output_size=self.num_embedding_channels,
          with_bias=self.with_bias,
          w_init=hk.initializers.VarianceScaling(1.0),
          name=f'embed_{modality_name}')
      out[modality_name] = conv_1d_embed(value)

    return out

  def unembed(self, inputs: Mapping[str,
                                    chex.Array]) -> Dict[str, chex.Array]:
    """Reverses an embed operation, reproducing the shape of original inputs.

    Args:
      inputs: A dictionary of modality name and (batch, index,
        num_embedding_channels).

    Returns:
      A dictionary of modality name and (batch, index, num_embedding_channels).
    """
    _assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
    assert self._orig_channels is not None, 'Must call embed() first.'
    assert (
        list(inputs.keys()) == list(self._orig_channels.keys())
    ), f'Modality names must be consistent. Expected {self._orig_channels.keys()}; found {inputs.keys()}.'

    out = {}
    for modality_name, value in inputs.items():
      # value: (batch, index, channels)

      conv_1d_unembed = hk.Linear(
          output_size=self._orig_channels[modality_name],
          with_bias=self.with_bias,
          w_init=hk.initializers.VarianceScaling(0.0),
          name=f'unembed_{modality_name}')
      out[modality_name] = conv_1d_unembed(value)

    return out


class PositionEncoder(hk.Module):
  """Adds position encodings to input channels.

  Inputs should be a dictionary of {modality_name:
  (batch_size, index_dim, num_channels)}. The output format will be identical.

  Note both inputs and outputs are ungrouped. Grouping is handled by the
  Grouper module.
  """

  def __init__(
      self,
      num_position_encoding_channels: Optional[int] = None,
      name: str = 'position_encoder',
  ):
    super().__init__(name=name)
    self.num_position_encoding_channels = num_position_encoding_channels

  def __call__(
      self, inputs: Mapping[str, chex.Array]
  ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array]]:
    """Adds position encodings to the inputs and also returns the encodings.

    Args:
      inputs: A dictionary of {modality_name:
        (batch_size, index_dim, num_channels)} inputs.

    Returns:
      A tuple of the inputs with added position encodings, as well as the
      raw encodings.
    """
    _assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
    num_channels = next(iter(inputs.values())).shape[CHANNELS_DIM]

    if self.num_position_encoding_channels is None:
      num_position_encoding_channels = num_channels
    else:
      num_position_encoding_channels = self.num_position_encoding_channels

    out = {}
    pos_encodings = {}
    for modality_name, value in inputs.items():
      # value: (batch, index, channels)
      pos_encodings_i = perceiver_helpers.TrainablePositionEncoding(
          index_dim=value.shape[INDEX_DIM],
          num_channels=value.shape[CHANNELS_DIM],
          # Unshared between modalities.
          name=f'pos_emb_mae_{modality_name}')(value.shape[BATCH_DIM])

      if num_position_encoding_channels != num_channels:
        # Project to required size.
        conv_1d_encode = hk.Linear(
            output_size=num_channels,
            with_bias=False,
            w_init=hk.initializers.VarianceScaling(1.0),
            # Shared between modalities.
            name='position_encoding_linear')
        pos_encodings_i = conv_1d_encode(pos_encodings_i)

      pos_encodings[modality_name] = pos_encodings_i
      out[modality_name] = value + pos_encodings[modality_name]

    return out, pos_encodings


@chex.dataclass
class _GroupInfo:
  modality_name: str
  group_idx: List[int]
  final_padding: int
  group_padding: int


class ConstNumGrouper(hk.Module):
  """Groups inputs into a constant number of groups.

  The group size will grow based on the inputs.

  Inputs should be a dictionary of {modality_name:
  (batch_size, index_dim, num_channels)}. The output format will be (batch_size,
  num_groups, new_index_dim (computed), num_channels).

  Notes: Inputs will be ordered based on the insertion order of the dict. Make
  sure this is consistent across calls. batch_size and num_channels must be
  constant across modalities.

  The Grouper will ensure that multiple modalities are not mixed in a single
  group. Padding will be added proportionally at the end of each group, with
  extra padding at the last group of each modality.
  """

  def __init__(
      self,
      *,
      num_groups: int,
      name: str = 'constant_number_grouper',
  ):
    """Builds the Grouper.

    Args:
      num_groups: The number of groups to create.
      name: Haiku module name.
    """

    super().__init__(name=name)
    self.num_groups = num_groups
    self._group_map = None

  def _build_group_map(self, inputs: Mapping[str, chex.Array]):
    index_dims = [v.shape[INDEX_DIM] for v in inputs.values()]
    assign_groups_to_modalities = perceiver_helpers.assign_groups_to_modalities
    num_groups_per_modality, index_dim_per_group = assign_groups_to_modalities(
        self.num_groups, index_dims)

    group_map = []
    next_group_id = 0
    for (name, value), num_modality_groups in zip(inputs.items(),
                                                  num_groups_per_modality):
      index_dim = value.shape[INDEX_DIM]
      assigned_groups = list(
          range(next_group_id, next_group_id + num_modality_groups))
      next_group_id += num_modality_groups

      final_padding = perceiver_helpers.padding_to_make_divisible(
          index_dim, num_modality_groups)
      local_index_dim_per_group = (index_dim +
                                   final_padding) // num_modality_groups
      group_padding = index_dim_per_group - local_index_dim_per_group

      group_map.append(
          _GroupInfo(
              modality_name=name,
              group_idx=assigned_groups,
              final_padding=final_padding,
              group_padding=group_padding))

    self._group_map = group_map

  def group(self, inputs: Mapping[str, chex.Array]) -> chex.Array:
    """Groups a given input with the appropriate padding.

    This method can be called multiple times on inputs that require similar
    grouping and padding (e.g., a sample and its attention mask).

    Args:
      inputs: A dict of modality names and (batch, index, channel) values.

    Returns:
      A tensor of shape (batch, group, index, channel).
    """

    _assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
    self._build_group_map(inputs)

    grouped_inputs = []
    for group_info, value in zip(self._group_map, inputs.values()):
      x = jnp.pad(value, ((0, 0), (0, group_info.final_padding), (0, 0)))

      x = einshape('b(gm)...->bgm...', x, g=len(group_info.group_idx))
      x = jnp.pad(x, ((0, 0), (0, 0), (0, group_info.group_padding), (0, 0)))
      grouped_inputs.append(x)

    return jnp.concatenate(grouped_inputs, axis=1)

  def ungroup(self, latents: chex.Array) -> Dict[str, chex.Array]:
    """Ungroups a given input into a dict of modalities and values.

    Args:
      latents: A tensor of (batch, group, index, channel).

    Returns:
      A dict of the original modality names and their values.
    """

    assert len(latents.shape) == 4
    out = {}

    for group_info in self._group_map:
      # Select only the relevant groups.
      x = latents[:, group_info.group_idx, :, :]
      # Remove per-group padding.
      x = x[:, :, :x.shape[INDEX_DIM] - group_info.group_padding, :]
      x = einshape('bgm...->b(gm)...', x, g=len(group_info.group_idx))
      # Remove final padding.
      x = x[:, :x.shape[INDEX_DIM] - group_info.final_padding, :]

      out[group_info.modality_name] = x

    return out


class ConcatenateGrouper(hk.Module):
  """Concatenates inputs into a single group, shared across all modalities.

  Inputs should be a dictionary of {modality_name:
  (batch_size, index_dim, num_channels)}. The output format will be (batch_size,
  num_groups, total_index_dim, num_channels), where
  total_index_dim = sum_{modality_i} index_dim_i.

  Notes: Inputs will be ordered based on the insertion order of the dict. Make
  sure this is consistent across calls. batch_size and num_channels must be
  constant across modalities.
  """

  def __init__(
      self,
      *,
      name: str = 'concatenate_grouper',
  ):
    """Builds the Grouper.

    Args:
      name: Haiku module name.
    """

    super().__init__(name=name)
    self._index_dims = None
    self._input_names = None

  def group(self, inputs: Mapping[str, chex.Array]) -> chex.Array:
    """Groups a given input.

    This method can be called multiple times on inputs that require similar
    grouping (e.g., a sample and its attention mask).

    Args:
      inputs: A dict of modality names and (batch, index, channel) values.

    Returns:
      A tensor of shape (batch, group, index, channel).
    """

    _assert_input_shapes(inputs, expected_rank=3, constant_channels=True)
    self._input_names = inputs.keys()
    self._index_dims = [v.shape[INDEX_DIM] for v in inputs.values()]

    # [B, (I_0 + I_1 + ... + I_N), C]
    grouped = jnp.concatenate(list(inputs.values()), axis=1)
    # Add a dummy group axis:
    return grouped[:, None, ...]

  def ungroup(self, latents: chex.Array) -> Dict[str, chex.Array]:
    """Ungroups a given input into a dict of modalities and values.

    Args:
      latents: A tensor of (batch, group, index, channel).

    Returns:
      A dict of the original modality names and their values.
    """

    assert len(latents.shape) == 4

    start_idx = 0
    out = dict()
    for name, index_dim in zip(self._input_names, self._index_dims):
      end_idx = start_idx + index_dim
      # [B, 1, (I_0 + I_1 + ... + I_N), C] -> [B, I_i, C]
      out[name] = latents[:, 0, start_idx:end_idx, :]
      start_idx = end_idx

    return out


class ReconstructionHead(hk.Module):
  """Produces a reconstruction from perceiver latents and an MAE query.

  The reconstruction is in a grouped and embedded form, similar to the input
  of a PerceiverBlock. It needs to be ungrouped and unembedded.
  """

  def __init__(self,
               *,
               use_post_attention_residual: bool = False,
               name: str = RECONSTRUCTION_HEAD_NAME):
    super().__init__(name=name)
    self._use_post_attention_residual = use_post_attention_residual

  def __call__(self, latents: chex.Array, *, mae_query: chex.Array,
               is_training: bool) -> chex.Array:
    """Given latents and an MAE query, builds the reconstruction.

    Args:
      latents: The output of a PerceiverBlock.
      mae_query: MAE query - the second return value of PositionalEncoder.
      is_training: Current execution mode.

    Returns:
      A grouped array of reconstructions for the query. The array will have
      the shape of the MAE query.
    """

    chex.assert_rank(latents, 4)
    chex.assert_rank(mae_query, 4)

    projector = HiPCrossAttention(
        widening_factor=1,
        num_heads=1,
        use_post_attention_residual=self._use_post_attention_residual,
    )

    predictions = projector(
        inputs=latents, query_inputs=mae_query, is_training=is_training)

    return predictions


def _assert_input_shapes(inputs: Mapping[str, chex.Array],
                         *,
                         expected_rank: int,
                         constant_channels: bool = False):
  """Given an inputs dictionary, asserts all shapes are correct."""

  batch_size = None
  num_channels = None
  for modality_name, values in inputs.items():
    assert len(values.shape) == expected_rank
    if batch_size is None:
      batch_size = values.shape[BATCH_DIM]
      num_channels = values.shape[CHANNELS_DIM]
    else:
      assert (batch_size == values.shape[BATCH_DIM]
             ), f'batch size is inconsistent for input {modality_name}'
      if constant_channels:
        assert (num_channels == values.shape[-1]
               ), f'num channels is inconsistent for input {modality_name}'

  return batch_size, (num_channels if constant_channels else None)
