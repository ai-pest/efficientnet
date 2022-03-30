# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 Northern System Service Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Changelog
# 2020-10-01 v1
# * Add warmstart, transfer learning, keep_checkpoint_max, arcface,
#   resize_method
# 2020-10-05 v2
# * Add confusion matrix
# * Add transfer learning for efficientnet-b6
# * Add `--eval_iterator` CLI argument
# 2020-11-17 v3
# * Remove export()
# * Bugfix: Flag `--warm_start_path` was ignored in v2. Fixed.
# 2020-11-17 v4rc
# * Inspecting `--warm_start_path`. Reverted fix in v3.
# 2020-11-17 v4rc2
# * Bugfix: Stop restoring from warm_start_path when stage >= 2
#   (This will not affect model performance, since the latest ckpt is reloaded
#   at the start of stage 2 anyway.)
# 2020-11-18 v4rc3
# * Bugfix: Reactivate warm_start_path override.
#   (This will not affect model performance, since warm_start_path is set as
#   params dict is initialized)
# * `--eval_iterator=all` now runs eval in the order of steps
# * Remove `train_and_eval` mode and `async_checkpoint` for code clarity
# 2020-11-20 v4rc4
# * Add `--eval_iterator=[step]`
# * Fixed mysterious bug: build_model() got multiple values for argument
#   'model_name'
# * Add `--include_validation`
# 2022-01-31 v4rc5
# * Add `--num_model_classes` for situations where the dataset has less classes
#   than the model output
# * Add `--translate_classes` for JIT label translation

"""Train a EfficientNets on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2  # used for summaries only.
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from tensorflow.python.lib.io  import file_io

import imagenet_input
import model_builder_factory
import utils
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5,
    help=('The max number of recent checkpoint files to keep. Keeps all '
    'checkpoints if set to 0.'))

flags.DEFINE_string(
    'model_name',
    default='efficientnet-b0',
    help=('The model name among existing configurations.'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train", "eval"}.')

flags.DEFINE_string(
    'augment_name', default=None,
    help='`string` that is the name of the augmentation method'
         'to apply to the image. `autoaugment` if AutoAugment is to be used or'
         '`randaugment` if RandAugment is to be used. If the value is `None` no'
         'augmentation method will be applied applied. See autoaugment.py for  '
         'more details.')

flags.DEFINE_string(
    'augment_subpolicy', default=None,
    help='`string` specifying an augment subpolicy to be applied.'
         'Available only when augment_name is `autoaugment`.')

flags.DEFINE_integer(
    'randaug_num_layers', default=2,
    help='If RandAug is used, what should the number of layers be.'
         'See autoaugment.py for detailed description.')

flags.DEFINE_integer(
    'randaug_magnitude', default=10,
    help='If RandAug is used, what should the magnitude be. '
         'See autoaugment.py for detailed description.')

flags.DEFINE_integer(
    'input_image_size', default=None,
    help=('Input image size: it depends on specific model name.'))

flags.DEFINE_integer(
    'train_batch_size', default=2048, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'train_steps', default=None,
    help=('The number of steps to use for training. '
          'Defaults to None. When transfer_learning, use --transfer_schedule '
          'instead. (This flag will result in AssertionError in such cases.) '
          'Steps should be adjusted according to the --train_batch_size'
          'flag.'))

flags.DEFINE_list(
    'transfer_schedule', default=None,
    help=('Will perform transfer learning with a schedule defined in the '
          'passed list. Defaults to None. Trains the head block for l[0] '
          'EPOCHS (not steps!), then unfreezes all layers and continues. '
          'Training stops at l[1] epochs.'))

flags.DEFINE_string(
    'warm_start_path', default=None,
    help=('The path to the model to start transfer learning from.'))

flags.DEFINE_integer(
    'steps_per_eval', default=6255,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout', default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_string(
    'eval_iterator', default='latest',
    help=('The iterator to be used in eval mode (`latest`(default), `all` or '
          'step [int] to start evaluation from).'
          ' `latest` iterator scans only for the newest ckpts. `all` iterator'
          ' will yield all ckpts found in model_dir. [int] is the same as'
          ' `all`, except that evaluation starts from the step equal to or'
          ' greater than given.'))
flags.register_validator(
    'eval_iterator',
    lambda v: v in ('latest', 'all') or v.isdigit(),
    message='Unknown eval_iterator. Set to `latest`, `all` or step [int] to'
        ' start evaluation from.'
)

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=64,
    help=('Number of parallel threads in CPU for the input pipeline'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string(
    'bigtable_instance', None,
    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string(
    'bigtable_table', 'imagenet',
    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string(
    'bigtable_train_prefix', 'train_',
    'The prefix identifying training rows.')
flags.DEFINE_string(
    'bigtable_eval_prefix', 'validation_',
    'The prefix identifying evaluation rows.')
flags.DEFINE_string(
    'bigtable_column_family', 'tfexample',
    'The column family storing TFExamples.')
flags.DEFINE_string(
    'bigtable_column_qualifier', 'example',
    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))
flags.DEFINE_integer(
    'num_label_classes', default=1000,
    help='Number of classes in the dataset, at least 2')
flags.DEFINE_integer(
    'num_model_classes', default=None,
    help='Number of classes of the model head. If specified, the model will be'
    ' initialized as a `num_model_classes`-class classifier, but the metric'
    ' will respect `num_label_classes`.')

flags.DEFINE_float(
    'batch_norm_momentum',
    default=None,
    help=('Batch normalization layer momentum of moving average to override.'))
flags.DEFINE_float(
    'batch_norm_epsilon',
    default=None,
    help=('Batch normalization layer epsilon to override..'))

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_bfloat16',
    default=False,
    help=('Whether to use bfloat16 as activation for training.'))

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))
flags.DEFINE_bool(
    'export_to_tpu', default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=None,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_float(
    'survival_prob', default=None,
    help=('Drop connect rate for the network.'))

flags.DEFINE_float(
    'mixup_alpha',
    default=0.0,
    help=('Alpha parameter for mixup regularization, 0.0 to disable.'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                     'which the global step information is logged.')

flags.DEFINE_bool(
    'use_cache', default=False, help=('Enable cache for training input.'))

flags.DEFINE_float(
    'depth_coefficient', default=None,
    help=('Depth coefficient for scaling number of layers.'))

flags.DEFINE_float(
    'width_coefficient', default=None,
    help=('Width coefficient for scaling channel size.'))

flags.DEFINE_bool(
    'use_arcface', default=False, help=('Enable arcface.'))

flags.DEFINE_string(
    'resize_method', default=None,
    help=('Forces the model to resize images with the specified method '
          '(if provided.)'))

flags.DEFINE_bool(
    'include_validation', default=False,
    help=('if True, train with train and validation subsets.'))

flags.DEFINE_string(
    'translate_classes', default=None,
    help=('If a filepath to a JSON dicitonary is set, translates labels '
    'accordingly.'))


def model_fn(features, labels, mode, params):
  """The model_fn to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of one hot labels for the data samples, with shape
            `[batch, num_classes]`.
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']

  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])
    stats_shape = [3, 1, 1]
  else:
    stats_shape = [1, 1, 3]

  if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (FLAGS.moving_average_decay > 0)
  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)
  logging.info('Using open-source implementation.')
  override_params = {}
  if FLAGS.batch_norm_momentum is not None:
    override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
  if FLAGS.batch_norm_epsilon is not None:
    override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
  if FLAGS.dropout_rate is not None:
    override_params['dropout_rate'] = FLAGS.dropout_rate
  if FLAGS.survival_prob is not None:
    override_params['survival_prob'] = FLAGS.survival_prob
  if FLAGS.data_format:
    override_params['data_format'] = FLAGS.data_format
  # v4rc5: The model has `num_model_classes` heads, if specified.
  if FLAGS.num_model_classes:
    override_params['num_classes'] = FLAGS.num_model_classes
  else:
    override_params['num_classes'] = FLAGS.num_label_classes
  if FLAGS.depth_coefficient:
    override_params['depth_coefficient'] = FLAGS.depth_coefficient
  if FLAGS.width_coefficient:
    override_params['width_coefficient'] = FLAGS.width_coefficient
  # v4rc3: Reactivated warm_start_path override.
  if FLAGS.warm_start_path is not None:
    override_params['warm_start_path'] = FLAGS.warm_start_path
  if FLAGS.use_arcface:
    override_params['use_arcface'] = FLAGS.use_arcface

  model_name_to_blocks = {
      'efficientnet-b1': 23,
      'efficientnet-b6': 44,
      'efficientnet-b7': 54,
      'efficientnet-l2': 87
      }
  last_block = model_name_to_blocks[FLAGS.model_name]

  def normalize_features(features, mean_rgb, stddev_rgb):
    """Normalize the image given the means and stddevs."""
    features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
    features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
    return features

  def build_model():
    """Build model using the model_name given through the command line."""
    model_builder = model_builder_factory.get_model_builder(FLAGS.model_name)
    normalized_features = normalize_features(features, model_builder.MEAN_RGB,
                                             model_builder.STDDEV_RGB)
    print("model_name: {} ()".format(FLAGS.model_name))
    logits, _ = model_builder.build_model(
        normalized_features,
        model_name=FLAGS.model_name,
        training=is_training,
        override_params=override_params,
        model_dir=FLAGS.model_dir)
    return logits

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      logits = tf.cast(build_model(), tf.float32)
  else:
    logits = build_model()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  # v4rc5: Skip loss definition to avoid shape mismatch
  if is_training:
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        label_smoothing=FLAGS.label_smoothing)
    # Add weight decay to the loss for non-batch-normalization variables.
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])
  else:
    # v4rc5: A stub loss definition, since EstimatorSpec does not allow leaving
    # loss to None
    loss = tf.constant(0, dtype=tf.float32)

  global_step = tf.train.get_global_step()
  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  host_call = None
  restore_vars_dict = None
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])

    scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)
    logging.info('base_learning_rate = %f', FLAGS.base_learning_rate)
    learning_rate = utils.build_learning_rate(scaled_lr, global_step,
                                              params['steps_per_epoch'])
    ## DEBUG: Changed the optimizer from RMSProp to SGD, since published models
    ## are traind that way. See issue #652.
    optimizer = utils.build_optimizer(learning_rate, optimizer_name='sgd')
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Scheduled layer freeze/unfreeze for transfer learning.
    # As training proceeds, more layers become trainable.
    if params['transfer_learning']:
      assert params['stage'] is not None
      def get_layers_to_train(scope_name):
        '''Returns all variables in the given scope (e.g.
        'efficientnet-b1/head/') except batchnorm. Batchnorm layers should be
        left in their initial state. See this for details:
        https://keras.io/guides/transfer_learning/
        '''
        return [
            v for v in tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
            if 'batch_normalization' not in v.name]

      stage_to_trainable_layers = {}

      # Stage 1: Train head
      stage_to_trainable_layers[1] = \
        get_layers_to_train('{}/head/'.format(FLAGS.model_name))

      # Stage 2: Train all layers
      stage_to_trainable_layers[2] = stage_to_trainable_layers[1]
      for n in range(last_block, -1, -1):
        stage_to_trainable_layers[2] = \
          get_layers_to_train('{}/blocks_{}/'.format(FLAGS.model_name, n)) \
          + stage_to_trainable_layers[2]

      vars_to_train = stage_to_trainable_layers[params['stage']]

      logging.info(
        'Layers to train (Stage {}): {}'.format(
          params['stage'], stage_to_trainable_layers[params['stage']]))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, var_list=vars_to_train)

    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, lr, ce, lo):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf2.summary.create_file_writer(
            FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
          with tf2.summary.record_if(True):
            tf2.summary.scalar('learning_rate', lr[0], step=gs)
            tf2.summary.scalar('current_epoch', ce[0], step=gs)
            tf2.summary.scalar('custom_loss', lo[0], step=gs)

            return tf.summary.all_v2_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      gs_t = tf.reshape(global_step, [1])
      lr_t = tf.reshape(learning_rate, [1])
      ce_t = tf.reshape(current_epoch, [1])
      lo_t = tf.reshape(loss, [1])

      #host_call = (host_call_fn, [gs_t, lr_t, ce_t])
      host_call = (host_call_fn, [gs_t, lr_t, ce_t, lo_t])

  else:
    train_op = None
    if has_moving_average_decay:
      # Load moving average variables for eval.
      restore_vars_dict = ema.variables_to_restore(ema_vars)

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    path_to_id2label = \
      FLAGS.data_dir + 'synset_id2label.json' if FLAGS.data_dir.endswith('/') \
      else  FLAGS.data_dir + '/synset_id2label.json'
    print(' - Loading path_to_id2label ({})'.format(path_to_id2label))
    try:
      with file_io.FileIO(path_to_id2label, 'r') as f:
        id2label = json.load(f)
    except:
      id2label = None
    print(' - id2label: {}'.format(id2label))

    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch, num_classes]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      num_classes = FLAGS.num_label_classes     # Example:
      labels = tf.argmax(labels, axis=1)        # [4, 1, 1, 0, 1, ...]
      predictions = tf.argmax(logits, axis=1)   # [4, 1, 0, 0, 1, ...]

      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      ## Per-class metrics
      per_class_recall = {}
      per_class_precision = {}

      #for k in range(num_classes):                # say k = 1
      #  #class_name = id2label[str(k)] if id2label is not None
      #  #             else '{:02d}'.format(k)
      #  class_name = f'{k:02d}'
      #  ## Recall
      #  key = 'recall_{}'.format(class_name)
      #  per_class_recall[key] = tf.metrics.recall(
      #      labels=tf.equal(labels, k),           # [False, True, True,  ...]
      #      predictions=tf.equal(predictions, k)  # [False, True, False, ...]
      #  )
      #  ## Precision
      #  key = 'precision_{}'.format(class_name)
      #  per_class_precision[key] = tf.metrics.precision(
      #      labels=tf.equal(labels, k),
      #      predictions=tf.equal(predictions, k)
      #  )

      ## v2: Added confusion matrix
      cm = _streaming_confusion_matrix(
        labels=labels,
        predictions=predictions,
        num_classes=num_classes
      )

      return {
          #'id2label': id2label,
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
          'confusion_matrix': cm,
          **per_class_recall,
          **per_class_precision,
      }

    eval_metrics = (metric_fn, [labels, logits])

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  logging.info('number of trainable parameters: %d', num_params)
  logging.info('params: {}'.format(params))

  ## Define scaffold_fn
  if has_moving_average_decay and not is_training:
    # This scaffold is applied only for eval jobs.
    def _scaffold_fn():
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
    scaffold_fn = _scaffold_fn

  ## V1: added scaffold for transfer learning.
  ## Not sure about the assignment map though.
  elif is_training and (params['warm_start_path'] is not None):
    def _scaffold_fn():
      if FLAGS.model_name not in model_name_to_blocks:
        raise Exception(
          'Number of blocks for this model is not defined in scaffold_fn'
          '(yet). Could you add the last_block to model_name_to_blocks?')
      all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)
      vars_to_restore = {
        '{}/stem/'.format(FLAGS.model_name): \
          '{}/stem/'.format(FLAGS.model_name)}
      vars_to_restore.update({
        '{}/blocks_{}/'.format(FLAGS.model_name, n): \
          '{}/blocks_{}/'.format(FLAGS.model_name, n) \
        for n in range(last_block)
      })
      logging.info('all_vars: {}'.format(all_vars))
      logging.info('vars_to_restore: {}'.format(vars_to_restore))
      tf.train.init_from_checkpoint(params['warm_start_path'], vars_to_restore)
      return tf.train.Scaffold()
    scaffold_fn = _scaffold_fn

  else:
    scaffold_fn = None

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn
  )


def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError(
        'Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError(
        'Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project,
      'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(  # pylint: disable=g-complex-comprehension
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def pprint_metrics(metrics, id2label):
  """Pretty prints metrics.
  Args
    metrics: `dict` of metrics.
    id2label: `list` with signature [`str`, ..], with each element representing
      a label.
  Returns
    ret `str`, which is a prettified output of metrics.
  """
  ## Get values
  global_step = metrics['global_step']
  top_1_accuracy = metrics['top_1_accuracy']
  top_5_accuracy = metrics['top_5_accuracy']
  cm = metrics['confusion_matrix']
  print('Confusion matrix type: {}'.format(type(cm)))
  precision_keys = [k for k in metrics.keys() if k.startswith('precision_')]
  recall_keys = [k for k in metrics.keys() if k.startswith('recall_')]

  ## Format values
  precisions = ['{},{}'.format(k, metrics[k]) for k in precision_keys]
  precisions = '\n'.join(precisions)
  recalls = ['{},{}'.format(k, metrics[k]) for k in recall_keys]
  recalls = '\n'.join(recalls)

  try:
    confusion_matrix = pd.DataFrame(
      cm, index=id2label, columns=id2label, dtype=np.int64).to_csv()
  except ValueError:
    ## ValueError happens when num_model_classes is specified.
    ## Discarding labels will mitigate it.
    confusion_matrix = pd.DataFrame(cm, dtype=np.int64).to_csv()

  ## Print
  ret = '''----- EVAL RESULTS -----
global_step,{global_step}
top_1_accuracy,{top_1_accuracy}
top_5_accuracy,{top_5_accuracy}
{precisions}
{recalls}
----- CONFUSION MATRIX -----
{confusion_matrix}'''.format(
    global_step=global_step,
    top_1_accuracy=top_1_accuracy,
    top_5_accuracy=top_5_accuracy,
    precisions=precisions,
    recalls=recalls,
    confusion_matrix=confusion_matrix,
  )
  return ret


def main(unused_argv):
  if FLAGS.mode in ('train', 'train_and_eval'):
    assert ((not FLAGS.train_steps and FLAGS.transfer_schedule) or
            (FLAGS.train_steps and not FLAGS.transfer_schedule)), \
           'Do not define --train_steps when --transfer_schedule is set. ' \
           'When not, always set --train_steps.'

  if FLAGS.warm_start_path is not None:
    assert FLAGS.transfer_schedule, 'When warmstarting from ' \
    '--warm_start_path, always set --transfer_schedule too. Otherwise what ' \
    'is the point of transfer learning...'

  input_image_size = FLAGS.input_image_size
  if not input_image_size:
    input_image_size = model_builder_factory.get_model_input_size(
        FLAGS.model_name)

  # For imagenet dataset, include background label if number of output classes
  # is 1001
  include_background_label = (FLAGS.num_label_classes == 1001)

  if FLAGS.tpu or FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None

  # v4rc3: Removed async checkpointing feature
  #if FLAGS.use_async_checkpointing: save_checkpoints_steps = None
  #else: save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)
  save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)

  config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long
  # Initializes model parameters.
  params = dict(
      steps_per_epoch=FLAGS.num_train_images / FLAGS.train_batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      warm_start_path=FLAGS.warm_start_path,
      transfer_learning=(FLAGS.transfer_schedule is not None))
  # v2: Added resize_method
  if FLAGS.resize_method == 'None':
    resize_method = tf.image.ResizeMethod.BILINEAR
  elif FLAGS.resize_method == 'pad_and_resize':
    resize_method = 'pad_and_resize'
  else:
    raise Exception('resize_method is invalid. Use "None" or "pad_and_resize".')

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  def build_imagenet_input(is_training):
    """Generate ImageNetInput for training and eval."""
    if FLAGS.bigtable_instance:
      logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
      select_train, select_eval = _select_tables_from_flags()
      return imagenet_input.ImageNetBigtableInput(
          is_training=is_training,
          use_bfloat16=FLAGS.use_bfloat16,
          transpose_input=FLAGS.transpose_input,
          selection=select_train if is_training else select_eval,
          num_label_classes=FLAGS.num_label_classes,
          include_background_label=include_background_label,
          augment_name=FLAGS.augment_name,
          mixup_alpha=FLAGS.mixup_alpha,
          randaug_num_layers=FLAGS.randaug_num_layers,
          randaug_magnitude=FLAGS.randaug_magnitude,
          resize_method=resize_method)
    else:
      if FLAGS.data_dir == FAKE_DATA_DIR:
        logging.info('Using fake dataset.')
      else:
        logging.info('Using dataset: %s', FLAGS.data_dir)

      return imagenet_input.ImageNetInput(
          is_training=is_training,
          data_dir=FLAGS.data_dir,
          transpose_input=FLAGS.transpose_input,
          cache=FLAGS.use_cache and is_training,
          image_size=input_image_size,
          num_parallel_calls=FLAGS.num_parallel_calls,
          use_bfloat16=FLAGS.use_bfloat16,
          num_label_classes=FLAGS.num_label_classes,
          include_background_label=include_background_label,
          augment_name=FLAGS.augment_name,
          augment_subpolicy=FLAGS.augment_subpolicy,
          mixup_alpha=FLAGS.mixup_alpha,
          randaug_num_layers=FLAGS.randaug_num_layers,
          randaug_magnitude=FLAGS.randaug_magnitude,
          resize_method=resize_method,
          include_validation=FLAGS.include_validation,
          translate_classes=FLAGS.translate_classes)

  imagenet_train = build_imagenet_input(is_training=True)
  imagenet_eval = build_imagenet_input(is_training=False)

  if FLAGS.mode == 'eval':
    est = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        export_to_tpu=FLAGS.export_to_tpu,
        params=params)
    eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size
    # Run evaluation when there's a new checkpoint
    #ckpt_state_proto = tf.train.get_checkpoint_state(FLAGS.model_dir)
    #if ckpt_state_proto is not None:
    #  for ckpt in ckpt_state_proto.all_model_checkpoint_paths:
    #    logging.info('Starting to evaluate.')
    ##v2: Iterate over all checkpoints

    ## id2label を取得
    path_to_id2label = \
      FLAGS.data_dir + 'synset_id2label.json' if FLAGS.data_dir.endswith('/') \
      else FLAGS.data_dir + '/synset_id2label.json'
    print(' - Loading path_to_id2label ({})'.format(path_to_id2label))
    try:
      with file_io.FileIO(path_to_id2label, 'r') as f:
        id2label = json.load(f)
        id2label = [id2label[str(k)] for k in range(len(id2label))]
    except:
      id2label = None
    print(' - id2label: {}'.format(id2label))

    ## 評価を実施
    ## V2: Add eval_iterator operator
    ## eval_iterator を作成
    if FLAGS.eval_iterator == 'latest':
      eval_iterator = tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout)
    elif FLAGS.eval_iterator == 'all':
      eval_iterator = tf.io.gfile.glob(
        '{}/model.ckpt-*.meta'.format(FLAGS.model_dir.rstrip('/')))
      eval_iterator = [ckpt.rstrip('.meta') for ckpt in eval_iterator]
      # v4rc3: `--eval_iterator=all` now runs eval in order of steps
      sort_ckpt = lambda fpath: int(fpath.split('/')[-1].split('-')[-1])
      eval_iterator = sorted(eval_iterator, key=sort_ckpt)
    else:
      # v4rc4: `--eval_iterator=[int]` runs eval on ckpts greater than [int].
      assert FLAGS.eval_iterator.isdigit()
      eval_iterator = tf.io.gfile.glob(
        '{}/model.ckpt-*.meta'.format(FLAGS.model_dir.rstrip('/')))
      eval_iterator = [ckpt.rstrip('.meta') for ckpt in eval_iterator]
      ckpt2step = lambda fpath: int(fpath.split('/')[-1].split('-')[-1])
      eval_iterator = sorted(eval_iterator, key=ckpt2step)
      eval_iterator = [ckpt for ckpt in eval_iterator \
          if ckpt2step(ckpt) >= int(FLAGS.eval_iterator)]

    ## チェックポイント巡回
    for ckpt in eval_iterator:
      print(" -> Running evaluation on '{}'...".format(ckpt))
      try:
        eval_results = est.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        eval_result_pprinted = pprint_metrics(eval_results, id2label)
        logging.info('Eval results')
        logging.info(eval_result_pprinted)
        utils.archive_ckpt(eval_results, eval_results['top_1_accuracy'], ckpt)
        ## Export eval_result to GS
        export_to = '{}/eval_result_{:06d}.csv'.format(
          FLAGS.model_dir.rstrip('/'), eval_results['global_step'])
        with tf.io.gfile.GFile(export_to, mode='w') as f:
          f.write(eval_result_pprinted)
        logging.info('Exported eval_result to {}'.format(export_to))

      # This except block might be executed when your configuration is
      # erratic (e.g. model_name is -b6 but the model was trained as -b1).
      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)
    else:
      logging.error(
        'File "checkpoint" is not found in model_dir. Terminating eval.')
      raise IOError('File "checkpoint" is not found in model_dir.')

  elif FLAGS.mode == 'train':
    start_timestamp = time.time()  # This time will include compilation time

    ## Non-scheduled training (w/o freezing layers)
    if FLAGS.train_steps is not None:
      current_step = estimator._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)
      logging.info('Running non-scheduled training.')
      logging.info(
          'Training for %d steps (%.2f epochs in total). Current step %d.',
          FLAGS.train_steps, FLAGS.train_steps / params['steps_per_epoch'],
          current_step)

      logging.info('Training the model.')
      est = tf.estimator.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=config,
          train_batch_size=FLAGS.train_batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          export_to_tpu=FLAGS.export_to_tpu,
          params=params)

      hooks = []
      est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=hooks
      )

    ## Scheduled training (first train head, then fine-tune all layers)
    else:
      assert FLAGS.transfer_schedule is not None
      logging.info('Running scheduled training.')
      ## FIXME: stage_to_trainable_layers is defined at model_fn, so we cannot
      ## assert all([k in stage_to_trainable_layers.keys()
      ##             for k in stages_to_epoch.keys()])
      stages_to_epoch = {
          n+1: int(epoch) for n, epoch in enumerate(FLAGS.transfer_schedule)}
      logging.info('Stages_to_epoch: {}'.format(stages_to_epoch))

      for stage in stages_to_epoch:
        stage_last_step = \
          FLAGS.num_train_images * stages_to_epoch[stage] \
          // FLAGS.train_batch_size

        if stage_last_step == 0:
          logging.info('--------------------------------------------------')
          logging.info(
            'Stage {} has zero steps. Moving on to the next!!'.format(stage))
          logging.info('--------------------------------------------------')
          continue

        logging.info('--------------------------------------------------')
        logging.info(
          'Starting stage {} of training. We will move to the next stage at '
          'step {}.'.format(stage, stage_last_step))
        logging.info('--------------------------------------------------')
        current_step = estimator._load_global_step_from_checkpoint_dir(
            FLAGS.model_dir)
        logging.info('Current step: {}'.format(current_step))
        # Parameter setup
        params['stage'] = stage

        if stage >= 2:
          # Skip warmstart once the training has started
          params['warm_start_path'] = None

        # Train for up to stages_to_epoch[stage] epochs. FLAGS.train_steps is
        # ignored. At the end of the stage, a checkpoint will be written to
        # --model_dir.
        est = tf.estimator.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            export_to_tpu=FLAGS.export_to_tpu,
            params=params)

        est.train(
            input_fn=imagenet_train.input_fn,
            max_steps=stage_last_step
        )

        logging.info('Finished training up to step %d. Elapsed seconds %d.',
                     stage_last_step, int(time.time() - start_timestamp))
      logging.info(
        'Transfer learning schedule exhausted. Training marked as finished.')

  else:
    raise NotImplementedError(
      'Hey, {} mode is not supported.'.format(FLAGS.mode))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
