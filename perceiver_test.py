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

"""Tests for Perceiver IO and HiP."""

import unittest

from absl import logging
import chex
import haiku as hk
import jax
from ml_collections import config_dict
import numpy as np
from parameterized import parameterized
import perceiver
import perceiver_helpers


def mock_data():
  return {
      'imagenet': {
          'imagenet_image':
              np.random.random((2, 1024, 3)),
          'imagenet_label':
              np.random.randint(low=0, high=1,
                                size=(2, 32, 1)).astype(np.float32),
      },
      'audioset': {
          'audioset_audio':
              np.random.random((2, 128, 16)),
          'audioset_label':
              np.random.randint(low=0, high=1,
                                size=(2, 16, 16)).astype(np.float32),
      }
  }


SAMPLE_LABEL_MODALITIES = ('audioset_label', 'coco_labels', 'imagenet_label',
                           'jft_label', 'multi_nli_labels', 'objects365_labels')

DEFAULT_MODEL_KWARGS = config_dict.ConfigDict({
    # Canonical models:
    'PerceiverIO': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'activation_name': 'sq_relu',
        'dropout_prob': 0.0,
        'drop_path_rate': 0.0,
    },
    'HiP': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'regroup_type': 'reshape',
        'activation_name': 'sq_relu',
        'dropout_prob': 0.0,
        'drop_path_rate': 0.0,
        # Optional index dimension overrides:
    },
    'HiPClassBottleneck': {
        # The size of the raw ('latent') position encodings.
        # If != the embedding size, will be projected.
        'num_position_encoding_channels': 512,
        'regroup_type': 'reshape',
        'activation_name': 'sq_relu',
        'dropout_prob': 0.0,
        'drop_path_rate': 0.0,
        'label_modalities': SAMPLE_LABEL_MODALITIES,
    },
})


def select_mae_outputs(output_dict):
  trimmed_output = {}
  for dataset, dataset_outs in output_dict.items():
    mae_key = perceiver_helpers.ModelOutputKeys.INPUT_RECONSTRUCTION
    trimmed_output[dataset] = {k: v for k, v in dataset_outs[mae_key].items()}
  return trimmed_output


class PerceiverTest(unittest.TestCase):

  @parameterized.expand([
      # Standard HiP
      ('HiP', '16', 99_224_996),
      ('HiP', '256', 95_585_188),
      ('HiP', 'Mini', 18_709_668),
      ('HiPClassBottleneck', 'Mini', 21_114_788),

  ])
  def test_all_perceiver_models(
      self, model_base_name, model_variant_name, expected_num_params):
    """Test creating Perceiver-like models, as well as their parameter counts.

    Parameters are counted per key, with the expectation that shared bottleneck
    models will have more parameters, because the encoder and decoder are not
    shared.

    Args:
      model_base_name: The model's name. Corresponds to a class in `perceiver`.
      model_variant_name: The model's variant.
      expected_num_params: Expected number of parameters for the module.
    """
    rng = jax.random.PRNGKey(4)

    def haiku_fn(inputs):
      out = {}
      model = perceiver.build_perceiver(
          model_base_name=model_base_name,
          model_variant_name=model_variant_name,
          model_kwargs=DEFAULT_MODEL_KWARGS[model_base_name])
      for dataset_name, v in inputs.items():
        out[dataset_name] = model(dataset_name, v, is_training=True)

      return out

    inputs = mock_data()

    with chex.fake_jit():
      transformed = hk.transform_with_state(haiku_fn)
      params, state = transformed.init(rng, inputs)

      outputs, _ = transformed.apply(params, state, rng, inputs)

      outputs = select_mae_outputs(outputs)
      chex.assert_trees_all_equal_shapes(outputs, inputs)

      _, treedef = jax.tree_flatten(params)
      num_params = hk.data_structures.tree_size(params)

      # pylint: disable=g-generic-assert
      logging.info('Checking parameter counts...')
      self.assertEqual(
          num_params, expected_num_params,
          f'{treedef}: \nExpected {expected_num_params} params, '
          f'got {num_params} for model {model_base_name}, '
          f'variant {model_variant_name}'
      )
      # pylint: enable=g-generic-assert

  @parameterized.expand([
      # Supersample latents at eval time.
      ('HiP', 'Mini', None, 128),
      # Subsample latents at train time.
      ('HiP', 'Mini', 32, None),
      ('HiPClassBottleneck', 'Mini', None, 128),
      # Subsample latents at train time.
      ('HiPClassBottleneck', 'Mini', 32, None),
  ])
  def test_processor_index_train_eval(
      self,
      model_base_name,
      model_variant_name,
      processor_index_dim_train,
      processor_index_dim_eval):
    """Test HiP processor train-time and eval-time index dimension overrides.

    Args:
      model_base_name: The model's name. Corresponds to a class in `perceiver`.
      model_variant_name: The model's variant.
      processor_index_dim_train: Train-time index dimension override for the
        processor block.
      processor_index_dim_eval: Eval-time index dimension override for the
        processor block.
    """
    rng = jax.random.PRNGKey(4)

    def haiku_fn(inputs, is_training):
      out = {}
      model_kwargs = DEFAULT_MODEL_KWARGS[model_base_name]

      # Override the processor_index_dim_ config settings.
      with model_kwargs.unlocked():
        model_kwargs.processor_index_dim_train = processor_index_dim_train
        model_kwargs.processor_index_dim_eval = processor_index_dim_eval

      model = perceiver.build_perceiver(
          model_base_name=model_base_name,
          model_variant_name=model_variant_name,
          model_kwargs=model_kwargs)

      for dataset_name, v in inputs.items():
        out[dataset_name] = model(dataset_name, v, is_training=is_training)

      return out

    inputs = mock_data()

    with chex.fake_jit():
      transformed = hk.transform_with_state(haiku_fn)
      params, state = transformed.init(rng, inputs, is_training=True)

      # Run as training
      outputs_train, _ = transformed.apply(
          params, state, rng, inputs, is_training=True)

      # Run as eval
      outputs_eval, _ = transformed.apply(
          params, state, rng, inputs, is_training=False)

      outputs_train = select_mae_outputs(outputs_train)
      chex.assert_trees_all_equal_shapes(outputs_train, inputs)

      outputs_eval = select_mae_outputs(outputs_eval)
      chex.assert_trees_all_equal_shapes(outputs_eval, inputs)

  @parameterized.expand([
      # Supersample latents at eval time.
      ('PerceiverIO', 'io_mini', None, 256),
      # Subsample latents at train time.
      ('PerceiverIO', 'io_mini', 64, None),
  ])
  def test_z_index_train_eval(
      self,
      model_base_name,
      model_variant_name,
      z_index_dim_train,
      z_index_dim_eval):
    """Test train-time and eval-time index dimension overrides.

    Args:
      model_base_name: The model's name. Corresponds to a class in `perceiver`.
      model_variant_name: The model's variant.
      z_index_dim_train: Optional train-time index dimension override.
      z_index_dim_eval: Optional eval-time index dimension override.
    """
    rng = jax.random.PRNGKey(4)

    def haiku_fn(inputs, is_training):
      out = {}
      model_kwargs = DEFAULT_MODEL_KWARGS[model_base_name]

      # Override the `eval_index_widening_factor` config setting.
      with model_kwargs.unlocked():
        model_kwargs.z_index_dim_train = z_index_dim_train
        model_kwargs.z_index_dim_eval = z_index_dim_eval

      model = perceiver.build_perceiver(
          model_base_name=model_base_name,
          model_variant_name=model_variant_name,
          model_kwargs=model_kwargs)

      for dataset_name, v in inputs.items():
        out[dataset_name] = model(dataset_name, v, is_training=is_training)

      return out

    inputs = mock_data()

    with chex.fake_jit():
      transformed = hk.transform_with_state(haiku_fn)
      params, state = transformed.init(rng, inputs, is_training=True)

      # Run as training
      outputs_train, _ = transformed.apply(
          params, state, rng, inputs, is_training=True)

      # Run as eval
      outputs_eval, _ = transformed.apply(
          params, state, rng, inputs, is_training=False)

      outputs_train = select_mae_outputs(outputs_train)
      chex.assert_trees_all_equal_shapes(outputs_train, inputs)

      outputs_eval = select_mae_outputs(outputs_eval)
      chex.assert_trees_all_equal_shapes(outputs_eval, inputs)
