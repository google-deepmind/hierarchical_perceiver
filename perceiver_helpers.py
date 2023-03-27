# Copyright 2023 DeepMind Technologies Limited
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

"""Helpers for Perceiver IO and HiP construction."""

import enum
import math
from typing import Any, List, Optional, Sequence, Tuple

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np


@enum.unique
class ModelOutputKeys(str, enum.Enum):
  INPUT_RECONSTRUCTION = 'input_reconstruction'
  LATENTS = 'latents'


def padding_to_make_divisible(index_dim: int, num_groups: int) -> int:
  return num_groups * math.ceil(index_dim / num_groups) - index_dim


def conv_1d(
    output_channels: int,
    init_scale: float = 1.0,
    with_bias: bool = True,
    name: Optional[str] = None) -> hk.Linear:
  """A 1D convolution."""
  return hk.Linear(
      output_size=output_channels,
      with_bias=with_bias,
      w_init=hk.initializers.VarianceScaling(init_scale),
      name=name)


def f32_softmax(x: chex.Array) -> chex.Array:
  if x.dtype in [jnp.bfloat16, jnp.float16]:
    return jax.nn.softmax(x.astype(jnp.float32)).astype(x.dtype)
  else:
    return jax.nn.softmax(x)


def layer_norm(x: chex.Array, name: Optional[str] = None) -> jax.Array:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                      name=name)(x)


def get_activation(activation_name: str) -> Any:
  if activation_name == 'sq_relu':
    return lambda x: jax.nn.relu(x)**2
  else:
    return getattr(jax.nn, activation_name)


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
  """Computes multi-head attention using a query, key and value.

  ... indicates multiple batch / group dimensions.

  Args:
    q: Query with shape [..., q_indices, num_heads, head_dim].
    k: Key with shape [..., kv_indices, num_heads, head_dim].
    v: Value with shape [..., kv_indices, num_heads, head_dim].
    dropout_prob: dropout probability on the attention weights.
    attention_mask: Array of shape [..., q_indices, kv_indices] indicating
      which keys/vals each query can attend to.
  Returns:
    Output of the attention with shape [..., q_indices, hiddens]
  """
  num_head_channels = q.shape[-1]
  attention = jnp.einsum('...nhc,...mhc->...hnm', q, k)
  attention *= 1. / math.sqrt(num_head_channels)

  if attention_mask is not None:
    # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
    # left-padded sampling. For more, see the colab below.
    # //experimental/users/tycai/lrl/NINF_NaN_investigation.ipynb
    large_k = jnp.array(1e4 if attention.dtype == jnp.float16 else 1e30,
                        dtype=attention.dtype)
    attention = jnp.where(
        # Add a dummy head dimension to the attention mask.
        attention_mask[..., None, :, :],
        attention,
        -large_k)

  normalized = f32_softmax(attention)
  if dropout_prob > 0:
    normalized = hk.dropout(hk.next_rng_key(), dropout_prob, normalized)
  summed = jnp.einsum('...hnm,...mhd->...nhd', normalized, v)

  # Concatenate heads:
  summed = einshape('...nhd->...n(hd)', summed)

  if attention_mask is not None:
    # Zero out the output of queries that attend to no keys or values.
    # -> [..., q_indices, 1]
    wipe_attn = jnp.all(attention_mask == 0, axis=-1, keepdims=True)
    summed = jnp.where(wipe_attn, jnp.zeros_like(summed), summed)
  return summed


def assign_groups_to_modalities(
    num_groups: int, index_dim_per_modality: Sequence[int]
) -> Tuple[List[int], int]:
  """Computes the number of groups assigned to each modality."""
  num_modalities = len(index_dim_per_modality)
  if num_modalities > num_groups:
    raise ValueError(
        f'{num_modalities} > {num_groups}.'
        'Can\'t yet deal with groups that have '
        'multiple modalities.')
  extra_groups = num_groups - num_modalities
  # Assign extra groups to each modality proportionally to the number of points
  # it contains (i.e. its index dimension). We do this by greedily assigning
  # groups to each modality so that all groups are used and the largest number
  # of points assigned to any group is minimized.
  num_groups_per_modality = [1] * num_modalities
  index_dim_per_group = list(index_dim_per_modality)
  for _ in range(extra_groups):
    modality = np.argmax(index_dim_per_group)
    num_groups_per_modality[modality] += 1
    index_dim_per_group[modality] = (
        index_dim_per_modality[modality] / num_groups_per_modality[modality])
  index_dim_per_group = math.ceil(max(index_dim_per_group))
  return num_groups_per_modality, index_dim_per_group


class TrainablePositionEncoding(hk.Module):
  """Trainable position encoding."""

  def __init__(self,
               index_dim: int,
               num_channels: int = 128,
               init_scale: float = 1.0,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._index_dim = index_dim
    self._num_channels = num_channels
    self._init_scale = init_scale

  def __call__(self,
               batch_size: Optional[int]) -> jnp.ndarray:
    pos_embs = hk.get_parameter(
        'pos_embs', [self._index_dim, self._num_channels],
        init=hk.initializers.VarianceScaling(scale=self._init_scale))

    if batch_size is not None:
      pos_embs = jnp.broadcast_to(
          pos_embs[None, :, :], (batch_size,) + pos_embs.shape)
    return pos_embs


class StochasticDepth(hk.Module):
  """Batchwise Dropout used in EfficientNet/NfNet, optionally sans rescaling."""

  def __init__(self,
               drop_rate: float,
               scale_by_keep: bool = False,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.drop_rate = drop_rate
    self.scale_by_keep = scale_by_keep

  def __call__(self, x: chex.Array, is_training: bool) -> jnp.ndarray:
    if not is_training:
      return x  # pytype: disable=bad-return-type  # numpy-scalars
    batch_size = x.shape[0]
    r = jax.random.uniform(
        hk.next_rng_key(),
        [batch_size] + [1] * (x.ndim - 1),
        dtype=x.dtype)
    keep_prob = 1. - self.drop_rate
    binary_tensor = jnp.floor(keep_prob + r)
    if self.scale_by_keep:
      x = x / keep_prob
    return x * binary_tensor


class Dense(hk.Module):
  """A Transformer-style dense module to follow attention."""

  def __init__(self,
               widening_factor: int = 4,
               dropout_prob: float = 0.0,
               init_scale: float = 1.,
               activation_name: str = 'sq_relu',
               name: Optional[str] = None):
    super().__init__(name=name)
    self._widening_factor = widening_factor
    self._dropout_prob = dropout_prob
    self._init_scale = init_scale
    self._activation_name = activation_name

  def __call__(self, x: chex.Array, is_training: bool = True) -> chex.Array:
    dropout_prob = self._dropout_prob if is_training else 0.0
    output_channels = x.shape[-1]
    x = conv_1d(
        output_channels=self._widening_factor * output_channels,
        init_scale=self._init_scale,
        name='mlp_hidden_linear')(x)
    x = get_activation(self._activation_name)(x)
    x = conv_1d(
        output_channels=output_channels,
        init_scale=self._init_scale,
        name='mlp_output_linear')(x)
    return hk.dropout(hk.next_rng_key(), dropout_prob, x)


class Attention(hk.Module):
  """Multi-headed {cross, self}-attention."""

  def __init__(self,
               num_heads: int = 8,
               init_scale: float = 1.0,
               with_final_bias: bool = True,
               dropout_prob: float = 0.0,
               qk_channels: Optional[int] = None,
               v_channels: Optional[int] = None,
               output_channels: Optional[int] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._init_scale = init_scale
    self._with_final_bias = with_final_bias
    self._dropout_prob = dropout_prob

    # If none of these are passed, the Q input determines the output shape:
    self._qk_channels = qk_channels
    self._v_channels = v_channels
    self._output_channels = output_channels

  def __call__(self, inputs_q, inputs_kv, attention_mask=None):
    # Q and K must have the same number of channels.
    # Default to preserving Q's input's shape.
    if self._qk_channels is None:
      self._qk_channels = inputs_q.shape[-1]
    # V's num_channels determines the shape of the output of QKV-attention.
    # Default to the same number of channels used in the key-query operation.
    if self._v_channels is None:
      self._v_channels = self._qk_channels
    # Project the output of QKV attention to a desired number of channels.
    # Default to the same number as the output of the QKV attention operation.
    if self._output_channels is None:
      self._output_channels = self._v_channels

    assert self._qk_channels % self._num_heads == 0
    assert self._v_channels % self._num_heads == 0
    qk_channels_per_head = self._qk_channels // self._num_heads
    v_channels_per_head = self._v_channels // self._num_heads

    # Project QKV to a common feature dimension.
    q = conv_1d(
        self._qk_channels,
        init_scale=self._init_scale,
        name='query_linear')(inputs_q)
    k = conv_1d(
        self._qk_channels,
        init_scale=self._init_scale,
        name='key_linear')(inputs_kv)
    v = conv_1d(
        self._v_channels,
        init_scale=self._init_scale,
        name='value_linear')(inputs_kv)

    # Reshape channels for multi-head attention.
    q = einshape('...m(hc)->...mhc', q,
                 h=self._num_heads, c=qk_channels_per_head)
    k = einshape('...n(hc)->...nhc', k,
                 h=self._num_heads, c=qk_channels_per_head)
    v = einshape('...n(hd)->...nhd', v,
                 h=self._num_heads, d=v_channels_per_head)

    result = attend(q, k, v, dropout_prob=self._dropout_prob,
                    attention_mask=attention_mask)

    return conv_1d(
        self._output_channels,
        with_bias=self._with_final_bias,
        init_scale=self._init_scale,
        name='attention_output_linear')(result)
