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

"""HiP and Perceiver IO model templates."""

import sys
from typing import Any, Dict, Mapping, Optional, Sequence

from absl import logging
import chex
import haiku as hk
from jax import numpy as jnp

from hierarchical_perceiver import perceiver_blocks
from hierarchical_perceiver import perceiver_helpers

PERCEIVER_MODULE_NAME = 'perceiver'


# Perceiver model variants.
VARIANTS = {
    'Mini': {
        'num_groups': (16, 1, 16),
        'num_self_attends_per_block': (2, 1, 1),
        'z_index_dim': (128, 64, 128),
        'num_z_channels': (128, 1024, 128),
        'num_cross_attend_heads': (1, 1, 1),
        'num_self_attend_heads': (4, 32, 4),
        'cross_attend_widening_factor': (1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4),
        'num_embedding_channels': 32,
    },
    '16': {
        'num_groups': (16, 4, 1, 1, 1, 4, 16),
        'num_self_attends_per_block': (2, 2, 18, 2, 1, 1, 1),
        'z_index_dim': (128, 256, 256, 64, 256, 256, 128),
        'num_z_channels': (128, 256, 512, 1024, 512, 256, 128),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (4, 8, 16, 32, 16, 8, 4),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 32,
    },
    '256': {
        'num_groups': (256, 64, 16, 4, 1, 1, 1, 4, 16, 64, 256),
        'num_self_attends_per_block': (1, 1, 2, 2, 18, 2, 1, 1, 1, 1, 1),
        'z_index_dim': (32, 64, 128, 256, 256, 64, 256, 256, 128, 64, 32),
        'num_z_channels': (64, 96, 128, 256, 512, 1024, 256, 128, 64, 32, 16),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 16,
    },
    '16x3': {
        'num_groups': (16, 1, 16),
        'num_self_attends_per_block': (2, 18, 2),
        'z_index_dim': (128, 256, 128),
        'num_z_channels': (128, 1024, 128),
        'num_cross_attend_heads': (1, 1, 1),
        'num_self_attend_heads': (4, 32, 4),
        'cross_attend_widening_factor': (1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4),
        'num_embedding_channels': 32,
    },
    # Perceiver IO
    'io_mini': {
        'num_self_attends_per_block': 2,
        'z_index_dim': 128,
        'num_z_channels': 128,
        'num_cross_attend_heads': 1,
        'num_self_attend_heads': 2,
        'cross_attend_widening_factor': 1,
        'self_attend_widening_factor': 2,
        'num_embedding_channels': 128,
    },
    'io_c_50m': {
        'num_self_attends_per_block': 8,
        'z_index_dim': 1024,
        'num_z_channels': 512,
        'num_cross_attend_heads': 8,
        'num_self_attend_heads': 8,
        'cross_attend_widening_factor': 4,
        'self_attend_widening_factor': 4,
        'num_embedding_channels': 512,
    },
    'io_c_150m': {
        'num_self_attends_per_block': 12,
        'z_index_dim': 1024,
        'num_z_channels': 896,
        'num_cross_attend_heads': 16,
        'num_self_attend_heads': 16,
        'cross_attend_widening_factor': 4,
        'self_attend_widening_factor': 4,
        'num_embedding_channels': 896,
    },
}


def _check_and_get_processor_idx(num_groups: Sequence[int]) -> int:
  # The processor is the central block in a HiP.
  # [enc_1, ..., enc_N, processor, dec_1, ..., dec_N]
  processor_idx = len(num_groups) // 2
  # The processor block has 1 group: it is essentially a Perceiver IO.
  assert num_groups[processor_idx] == 1, 'The processor must use 1 group.'
  return processor_idx


class PerceiverIO(hk.Module):
  """Perceiver IO.

  Perceiver IO is an anymodal, fully permutation-invariant model. It takes in
  a (usually large) input sequence, maps them to a (smaller) sequence with
  latent cross-attention, processes them with a homogeneous latent Transformer,
  then maps them to a (usually large) output sequence again with latent
  cross-attention. See https://arxiv.org/abs/2107.14795 for more details.

  For compatibility with HiP, this Perceiver IO includes a singleton group
  dimension: inputs are concatenated and newaxis'd to [B, 1, M, C] before
  processing (where M is the summed index dim of all input modalities).
  """

  def __init__(
      self,
      # Variant-specific hyperparams
      num_self_attends_per_block: int,
      z_index_dim: int,
      num_z_channels: int,
      num_cross_attend_heads: int,
      num_self_attend_heads: int,
      cross_attend_widening_factor: int,
      self_attend_widening_factor: int,
      num_embedding_channels: int,
      *,
      # Shared hyperparameters
      num_position_encoding_channels: Optional[int] = None,
      activation_name: str = 'sq_relu',
      z_index_dim_train: Optional[int] = None,
      z_index_dim_eval: Optional[int] = None,
      dropout_prob: float = 0.0,
      drop_path_rate: float = 0.0,
      name: str = PERCEIVER_MODULE_NAME,
  ):
    """Constructs the model.

    Args:
      num_self_attends_per_block: The number of self-attention layers in each
        block.
      z_index_dim: The number of latents in each block.
      num_z_channels: The number of channels in each block.
      num_cross_attend_heads: The number of heads in cross-attention layers in
        each block.
      num_self_attend_heads: The number of heads in self-attention layers in
        each block.
      cross_attend_widening_factor: The MLP channel widening factor in
        cross-attention layers in each block.
      self_attend_widening_factor: The MLP channel widening factor in
        self-attention layers in each block.
      num_embedding_channels: The number of channels used to embed inputs to and
        outputs from the model. Data from all modalities are projected to
        `num_embedding_channels`.
      num_position_encoding_channels: The number of channels of the raw position
        encoding. If num_position_encoding_channels != num_embedding_channels,
        position encodings are projected before adding to embedded inputs.
      activation_name: Activation for HiPCrossAttention and SelfAttention.
      z_index_dim_train: Optional train-time index dimension override.
      z_index_dim_eval: Optional eval-time index dimension override.
      dropout_prob: SelfAttention dropout probability.
      drop_path_rate: SelfAttention drop path rate.
      name: Haiku module name.
    """
    super().__init__(name=name)

    # Variant-specific hyperparams
    self.num_self_attends_per_block = num_self_attends_per_block
    self.z_index_dim = z_index_dim
    self.num_z_channels = num_z_channels
    self.num_cross_attend_heads = num_cross_attend_heads
    self.num_self_attend_heads = num_self_attend_heads
    self.cross_attend_widening_factor = cross_attend_widening_factor
    self.self_attend_widening_factor = self_attend_widening_factor
    self.num_embedding_channels = num_embedding_channels

    # Shared hyperparameters
    self.num_position_encoding_channels = num_position_encoding_channels
    self.activation_name = activation_name
    self.z_index_dim_train = z_index_dim_train
    self.z_index_dim_eval = z_index_dim_eval
    self.dropout_prob = dropout_prob
    self.drop_path_rate = drop_path_rate

  def __call__(self, dataset_name: str, inputs: Mapping[str, chex.Array], *,
               is_training: bool) -> Dict[str, chex.Array]:
    """Computes a reconstruction of the inputs through the model.

    Args:
      dataset_name: The name of the dataset (ignored).
      inputs: A dictionary of modality_name: value.
      is_training: Is this a training step.

    Returns:
      The computed output.
    """

    grouper = perceiver_blocks.ConcatenateGrouper()
    embedder = perceiver_blocks.Embedder(
        num_embedding_channels=self.num_embedding_channels)

    z_0 = embedder.embed(inputs)
    z, mae_query = perceiver_blocks.PositionEncoder(
        num_position_encoding_channels=self.num_position_encoding_channels,
    )(z_0)

    z = grouper.group(z)
    mae_query = grouper.group(mae_query)

    z = perceiver_blocks.PerceiverBlock(
        num_output_groups=1,
        output_index_dim=self.z_index_dim,
        num_output_channels=self.num_z_channels,
        num_self_attend_layers=self.num_self_attends_per_block,
        num_self_attend_heads=self.num_self_attend_heads,
        self_attend_widening_factor=self.self_attend_widening_factor,
        num_cross_attend_heads=self.num_cross_attend_heads,
        cross_attend_widening_factor=self.cross_attend_widening_factor,
        # Perceiver IO always uses a single group.
        regroup_inputs=False,
        regroup_type='',  # Ignored
        activation_name=self.activation_name,
        output_index_dim_train=self.z_index_dim_train,
        output_index_dim_eval=self.z_index_dim_eval,
        dropout_prob=self.dropout_prob,
        drop_path_rate=self.drop_path_rate,
        name='block_0')(z, is_training=is_training)

    reconstruction_z_out = perceiver_blocks.ReconstructionHead()(
        z, mae_query=mae_query, is_training=is_training)
    reconstruction_z_out = grouper.ungroup(reconstruction_z_out)
    reconstruction_output = embedder.unembed(reconstruction_z_out)

    z_out = grouper.ungroup(z)

    output_keys = perceiver_helpers.ModelOutputKeys
    return {  # pytype: disable=bad-return-type  # numpy-scalars
        output_keys.INPUT_RECONSTRUCTION: reconstruction_output,
        output_keys.LATENTS: z_out,
    }


class HiP(hk.Module):
  """Hierarchical Perceiver.

  See: https://arxiv.org/abs/2202.10890
  """

  def __init__(
      self,
      # Variant-specific hyperparams (e.g. for HiP-16, HiP-256)
      num_groups: Sequence[int],
      num_self_attends_per_block: Sequence[int],
      z_index_dim: Sequence[int],
      num_z_channels: Sequence[int],
      num_cross_attend_heads: Sequence[int],
      num_self_attend_heads: Sequence[int],
      cross_attend_widening_factor: Sequence[int],
      self_attend_widening_factor: Sequence[int],
      num_embedding_channels: int,
      *,
      # Shared hyperparameters
      num_position_encoding_channels: Optional[int] = None,
      regroup_type: str = 'reshape',
      activation_name: str = 'sq_relu',
      processor_index_dim_train: Optional[int] = None,
      processor_index_dim_eval: Optional[int] = None,
      dropout_prob: float = 0.0,
      drop_path_rate: float = 0.0,
      name: str = PERCEIVER_MODULE_NAME,
  ):
    """Constructs the model.

    Args:
      num_groups: The number of groups in each level of the HiP hierarchy.
      num_self_attends_per_block: The number of self-attention layers in each
        level of the HiP hierarchy.
      z_index_dim: The number of latents in each level of the HiP hierarchy.
      num_z_channels: The number of channels in each level of the HiP hierarchy.
      num_cross_attend_heads: The number of heads in cross-attention layers in
        each level of the HiP hierarchy.
      num_self_attend_heads: The number of heads in self-attention layers in
        each level of the HiP hierarchy.
      cross_attend_widening_factor: The MLP channel widening factor in
        cross-attention layers in each level of the HiP hierarchy.
      self_attend_widening_factor: The MLP channel widening factor in
        self-attention layers in each level of the HiP hierarchy.
      num_embedding_channels: The number of channels used to embed inputs to and
        outputs from the model. Data from all modalities are projected to
        `num_embedding_channels`.
      num_position_encoding_channels: The number of channels of the raw position
        encoding. If num_position_encoding_channels != num_embedding_channels,
        position encodings are projected before adding to embedded inputs.
      regroup_type: The regrouping strategy to use.
      activation_name: Activation for HiPCrossAttention and SelfAttention.
      processor_index_dim_train: Optional train-time index dimension override
        for the central processor block.
      processor_index_dim_eval: Optional eval-time index dimension override
        for the central processor block.
      dropout_prob: SelfAttention dropout probability.
      drop_path_rate: SelfAttention drop path rate.
      name: Haiku module name.
    """
    super().__init__(name=name)

    # Variant-specific hyperparams (e.g. for HiP-16, HiP-256)
    self.num_groups = num_groups
    self.num_self_attends_per_block = num_self_attends_per_block
    self.z_index_dim = z_index_dim
    self.num_z_channels = num_z_channels
    self.num_cross_attend_heads = num_cross_attend_heads
    self.num_self_attend_heads = num_self_attend_heads
    self.cross_attend_widening_factor = cross_attend_widening_factor
    self.self_attend_widening_factor = self_attend_widening_factor
    self.num_embedding_channels = num_embedding_channels

    # Shared hyperparameters
    self.num_position_encoding_channels = num_position_encoding_channels
    self.regroup_type = regroup_type
    self.activation_name = activation_name
    self.processor_index_dim_train = processor_index_dim_train
    self.processor_index_dim_eval = processor_index_dim_eval
    self.dropout_prob = dropout_prob
    self.drop_path_rate = drop_path_rate

    self.num_blocks = len(self.num_groups)

    assert self.num_blocks >= 3, (
        'At least 3 blocks are needed for U-Net residuals.')
    assert self.num_blocks % 2 == 1, (
        'HiP assumes an odd number of blocks: any number of paired '
        'encoder/decoder blocks plus 1 processor block.')
    self.processor_block_idx = _check_and_get_processor_idx(self.num_groups)

  def __call__(self, dataset_name: str, inputs: Mapping[str, chex.Array], *,
               is_training: bool) -> Dict[str, chex.Array]:
    """Computes a reconstruction of the inputs through the HiP.

    Args:
      dataset_name: The name of the dataset (ignored).
      inputs: A dictionary of modality_name: value.
      is_training: Is this a training step.

    Returns:
      The computed output.
    """

    grouper = perceiver_blocks.ConstNumGrouper(num_groups=self.num_groups[0])
    embedder = perceiver_blocks.Embedder(
        num_embedding_channels=self.num_embedding_channels)

    z_0 = embedder.embed(inputs)
    z, mae_query = perceiver_blocks.PositionEncoder(
        num_position_encoding_channels=self.num_position_encoding_channels,
    )(z_0)
    z = grouper.group(z)
    mae_query = grouper.group(mae_query)

    hidden_z = []
    for i in range(self.num_blocks):
      # UNet skips between corresponding encoder and decoder blocks.
      if i > self.processor_block_idx:
        pre_attention_residual = hidden_z[self.num_blocks - i - 1]
      else:
        pre_attention_residual = None

      if i == self.processor_block_idx:
        # Allow overrides of the number of processor-block latents.
        output_index_dim_train = self.processor_index_dim_train
        output_index_dim_eval = self.processor_index_dim_eval
      else:
        # Always use the default number of latents for encoder/decoder blocks.
        output_index_dim_train = None
        output_index_dim_eval = None

      z = perceiver_blocks.PerceiverBlock(
          num_output_groups=self.num_groups[i],
          output_index_dim=self.z_index_dim[i],
          num_output_channels=self.num_z_channels[i],
          num_self_attend_layers=self.num_self_attends_per_block[i],
          num_self_attend_heads=self.num_self_attend_heads[i],
          self_attend_widening_factor=self.self_attend_widening_factor[i],
          num_cross_attend_heads=self.num_cross_attend_heads[i],
          cross_attend_widening_factor=self.cross_attend_widening_factor[i],
          # The grouper takes care of the initial re-grouping.
          regroup_inputs=(i > 0),
          regroup_type=self.regroup_type,
          activation_name=self.activation_name,
          output_index_dim_train=output_index_dim_train,
          output_index_dim_eval=output_index_dim_eval,
          dropout_prob=self.dropout_prob,
          drop_path_rate=self.drop_path_rate,
          name=f'block_{i}')(
              z, is_training=is_training,
              pre_attention_residual=pre_attention_residual)
      hidden_z.append(z)

    reconstruction_z_out = perceiver_blocks.ReconstructionHead()(
        z, mae_query=mae_query, is_training=is_training)
    reconstruction_z_out = grouper.ungroup(reconstruction_z_out)
    reconstruction_output = embedder.unembed(reconstruction_z_out)

    z_out = grouper.ungroup(z)

    output_keys = perceiver_helpers.ModelOutputKeys
    return {  # pytype: disable=bad-return-type  # numpy-scalars
        output_keys.INPUT_RECONSTRUCTION: reconstruction_output,
        output_keys.LATENTS: z_out,
    }


class HiPClassBottleneck(hk.Module):
  """Hierarchical Perceiver with classes -> processor -> classes.

  This template handles class labels by passing them into and reading them out
  of the central processor block. All other modalities go through the encoder
  and decoder.

  See: https://arxiv.org/abs/2202.10890
  """

  def __init__(
      self,
      # Variant-specific hyperparams (e.g. for HiP-16, HiP-256)
      num_groups: Sequence[int],
      num_self_attends_per_block: Sequence[int],
      z_index_dim: Sequence[int],
      num_z_channels: Sequence[int],
      num_cross_attend_heads: Sequence[int],
      num_self_attend_heads: Sequence[int],
      cross_attend_widening_factor: Sequence[int],
      self_attend_widening_factor: Sequence[int],
      num_embedding_channels: int,
      label_modalities: Sequence[str],
      *,
      # Shared hyperparameters
      num_position_encoding_channels: Optional[int] = None,
      regroup_type: str = 'reshape',
      activation_name: str = 'sq_relu',
      processor_index_dim_train: Optional[int] = None,
      processor_index_dim_eval: Optional[int] = None,
      dropout_prob: float = 0.0,
      drop_path_rate: float = 0.0,
      name: str = PERCEIVER_MODULE_NAME):
    """Constructs the model.

    Args:
      num_groups: The number of groups in each level of the HiP hierarchy.
      num_self_attends_per_block: The number of self-attention layers in each
        level of the HiP hierarchy.
      z_index_dim: The number of latents in each level of the HiP hierarchy.
      num_z_channels: The number of channels in each level of the HiP hierarchy.
      num_cross_attend_heads: The number of heads in cross-attention layers in
        each level of the HiP hierarchy.
      num_self_attend_heads: The number of heads in self-attention layers in
        each level of the HiP hierarchy.
      cross_attend_widening_factor: The MLP channel widening factor in
        cross-attention layers in each level of the HiP hierarchy.
      self_attend_widening_factor: The MLP channel widening factor in
        self-attention layers in each level of the HiP hierarchy.
      num_embedding_channels: The number of channels used to embed inputs to and
        outputs from the model. Data from all modalities are projected to
        `num_embedding_channels`.
      label_modalities: The names of modalities to be passed in to the
        bottleneck.
      num_position_encoding_channels: The number of channels of the raw position
        encoding. If num_position_encoding_channels != num_embedding_channels,
        position encodings are projected before adding to embedded inputs.
      regroup_type: The regrouping strategy to use.
      activation_name: Activation for HiPCrossAttention and SelfAttention.
      processor_index_dim_train: Optional train-time index dimension override
        for the central processor block.
      processor_index_dim_eval: Optional eval-time index dimension override
        for the central processor block.
      dropout_prob: SelfAttention dropout probability.
      drop_path_rate: SelfAttention drop path rate.
      name: Haiku module name.
    """
    super().__init__(name=name)

    # Variant-specific hyperparams (e.g. for HiP-16, HiP-256)
    self.num_groups = num_groups
    self.num_self_attends_per_block = num_self_attends_per_block
    self.z_index_dim = z_index_dim
    self.num_z_channels = num_z_channels
    self.num_cross_attend_heads = num_cross_attend_heads
    self.num_self_attend_heads = num_self_attend_heads
    self.cross_attend_widening_factor = cross_attend_widening_factor
    self.self_attend_widening_factor = self_attend_widening_factor
    self.num_embedding_channels = num_embedding_channels

    # Shared hyperparameters
    self.num_position_encoding_channels = num_position_encoding_channels
    self.regroup_type = regroup_type
    self.activation_name = activation_name
    self.processor_index_dim_train = processor_index_dim_train
    self.processor_index_dim_eval = processor_index_dim_eval
    self.dropout_prob = dropout_prob
    self.drop_path_rate = drop_path_rate
    self.label_modalities = label_modalities

    self.num_blocks = len(self.num_groups)

    assert self.num_blocks >= 3, (
        'At least 3 blocks are needed for U-Net residuals.')
    assert self.num_blocks % 2 == 1, (
        'HiP assumes an odd number of blocks: any number of paired '
        'encoder/decoder blocks plus 1 processor block.')
    # Embedded class labels are input to and decoded from this block:
    self.processor_block_idx = _check_and_get_processor_idx(self.num_groups)

  def __call__(self, dataset_name: str, inputs: Mapping[str, chex.Array], *,
               is_training: bool) -> Dict[str, chex.Array]:
    """Computes a reconstruction of the inputs through the HiP.

    Args:
      dataset_name: The name of the dataset (ignored).
      inputs: A dictionary of modality_name: value.
      is_training: Is this a training step.

    Returns:
      The computed output.
    """
    grouper = perceiver_blocks.ConstNumGrouper(num_groups=self.num_groups[0])

    class_label_inputs = {k: v for k, v in inputs.items()
                          if k in self.label_modalities}
    inputs = {k: v for k, v in inputs.items()
              if k not in self.label_modalities}

    # Embed, position, and group the non-class-label inputs.
    embedder = perceiver_blocks.Embedder(
        num_embedding_channels=self.num_embedding_channels)
    z_0 = embedder.embed(inputs)

    z, mae_query = perceiver_blocks.PositionEncoder(
        num_position_encoding_channels=self.num_position_encoding_channels,
    )(z_0)
    z = grouper.group(z)
    mae_query = grouper.group(mae_query)

    num_blocks = len(self.num_groups)

    assert num_blocks >= 3, 'At least 3 blocks are needed for U-Net residuals.'

    hidden_z = []
    for i in range(num_blocks):
      # UNet skips between corresponding encoder and decoder blocks.
      if i > self.processor_block_idx:
        pre_attention_residual = hidden_z[num_blocks - i - 1]
      else:
        pre_attention_residual = None

      if i > 0:
        # Manually regroup the current latents to allow concatenation.
        # The grouper takes care of the initial regroup.
        z = perceiver_blocks.regroup(
            inputs=z,
            num_output_groups=self.num_groups[i],
            regroup_type=self.regroup_type)

      if i == self.processor_block_idx:
        mae_query_class = {}
        grouper_class = {}
        embedder_class = {}
        for k, v in class_label_inputs.items():
          # Concatenate the class inputs to the latents.
          assert z.shape[perceiver_blocks.GROUPS_DIM] == 1

          grouper_class[k] = perceiver_blocks.ConstNumGrouper(num_groups=1)
          # Embed and position encode class labels.
          embedder_class[k] = perceiver_blocks.Embedder(
              num_embedding_channels=z.shape[perceiver_blocks.CHANNELS_DIM])
          z_class = embedder_class[k].embed({k: v})

          z_class, mae_query_class[k] = perceiver_blocks.PositionEncoder(
              # Position encoding matches the embedding size.
              num_position_encoding_channels=z.shape[
                  perceiver_blocks.CHANNELS_DIM])(z_class)
          z_class = grouper_class[k].group(z_class)
          mae_query_class[k] = grouper_class[k].group(mae_query_class[k])

          z = jnp.concatenate([z, z_class], axis=perceiver_blocks.INDEX_DIM)

        # Allow overrides of the number of processor-block latents.
        output_index_dim_train = self.processor_index_dim_train
        output_index_dim_eval = self.processor_index_dim_eval
      else:
        # Always use the default number of latents for encoder/decoder blocks.
        output_index_dim_train = None
        output_index_dim_eval = None

      z = perceiver_blocks.PerceiverBlock(
          num_output_groups=self.num_groups[i],
          output_index_dim=self.z_index_dim[i],
          num_output_channels=self.num_z_channels[i],
          num_self_attend_layers=self.num_self_attends_per_block[i],
          num_self_attend_heads=self.num_self_attend_heads[i],
          self_attend_widening_factor=self.self_attend_widening_factor[i],
          num_cross_attend_heads=self.num_cross_attend_heads[i],
          cross_attend_widening_factor=self.cross_attend_widening_factor[i],
          # We've already re-grouped the latents: make sure they stay put!
          regroup_inputs=False,
          regroup_type=self.regroup_type,  # Ignored.
          activation_name=self.activation_name,
          output_index_dim_train=output_index_dim_train,
          output_index_dim_eval=output_index_dim_eval,
          dropout_prob=self.dropout_prob,
          drop_path_rate=self.drop_path_rate,
          name=f'perceiver_block_{i}')(
              z, is_training=is_training,
              pre_attention_residual=pre_attention_residual)
      hidden_z.append(z)

      if i == self.processor_block_idx:
        output_class = dict()
        rh_class = perceiver_blocks.ReconstructionHead()
        for k, v in mae_query_class.items():
          # Reconstruct the class-label inputs
          assert z.shape[perceiver_blocks.GROUPS_DIM] == 1  # pytype: disable=attribute-error  # numpy-scalars
          z_out_class = rh_class(z, mae_query=v, is_training=is_training)
          z_out_class = grouper_class[k].ungroup(z_out_class)
          output_class.update(embedder_class[k].unembed(z_out_class))

    reconstruction_z_out = perceiver_blocks.ReconstructionHead()(
        z, mae_query=mae_query, is_training=is_training)
    reconstruction_z_out = grouper.ungroup(reconstruction_z_out)
    reconstruction_output = embedder.unembed(reconstruction_z_out)
    # Merge class-label and non-class-label reconstructions into a single dict.
    reconstruction_output = {**reconstruction_output, **output_class}

    z_out = grouper.ungroup(z)

    output_keys = perceiver_helpers.ModelOutputKeys
    return {  # pytype: disable=bad-return-type  # numpy-scalars
        output_keys.INPUT_RECONSTRUCTION: reconstruction_output,
        output_keys.LATENTS: z_out,
    }


def build_perceiver(
    model_base_name: str,
    model_variant_name: Optional[str],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    searched_modules: Sequence[Any] = (sys.modules[__name__],),
) -> hk.Module:
  """Construct a Perceiver instance.

  Args:
    model_base_name: Name of a HiP-like base model class (e.g., 'HiP').
    model_variant_name: Name of a variant (e.g., '16'). Should be None for
      model classes with a baked-in variant (e.g. templates.HiPUnrolled16).
    model_kwargs: A dictionary of model kwargs. The key of the dictionary is a
      base model name (e.g., 'HiP') and the value is a kwargs dictionary.
    searched_modules: A list of modules to search for the given class.

  Returns:
    A constructed instance of the specified model.
  """
  candidate = None
  for module in searched_modules:
    if hasattr(module, model_base_name):
      candidate = getattr(module, model_base_name)
      break

  assert candidate is not None, (
      f'Failed to find class {model_base_name} in provided modules.')

  logging.info('Using Perceiver template: %s', model_base_name)
  if model_kwargs is None:
    model_kwargs = {}

  if model_variant_name is None:
    instance = candidate(
        **model_kwargs,
        name=PERCEIVER_MODULE_NAME)
  else:
    assert model_variant_name in VARIANTS, (
        f'VARIANTS does not contain {model_variant_name}. '
        'Please set variant to `None` if using a model with fixed variant.'
    )
    logging.info('Using Perceiver variant: %s', model_variant_name)

    instance = candidate(
        **model_kwargs,
        **VARIANTS[model_variant_name],
        name=PERCEIVER_MODULE_NAME)
  return instance
