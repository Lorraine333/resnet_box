# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.resnet import resnet_model
from official.resnet import eval
from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
# pylint: enable=g-bad-import-order


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    # print('examples per epoch', examples_per_epoch)
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    print('batches per epoch', total_batches)
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  # print(dataset)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()
  # return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):  # pylint: disable=unused-argument
    return model_helpers.generate_synthetic_data(
        input_shape=tf.TensorShape([batch_size, height, width, num_channels]),
        input_dtype=tf.float32,
        label_shape=tf.TensorShape([batch_size]),
        label_dtype=tf.int32)

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    return lr
  # def learning_rate_fn(global_step):
  #     return 0.0001

  return learning_rate_fn


def resnet_model_fn(input_features, input_labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False, model_method=1):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    fine_tune: If True only train the dense layers(final layers).

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  image_features = input_features['image']
  labels = input_labels['prob']
  # labels = tf.Print(labels, [labels], 'labels', summarize=100)
  tf.summary.image('images', image_features, max_outputs=6)


  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype)
  resnet_image_features = model.resnet_feature_call(input_features, mode == tf.estimator.ModeKeys.TRAIN)

  if model_method ==1:
      logits = model(input_features, resnet_image_features, labels)
  elif model_method == 2:
      print('joint model')
      logits = model.joint_model(input_features, resnet_image_features, labels)
  else:
      raise ValueError('invalid input for model method')


  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  # predictions = {
  #     'classes': tf.argmax(logits, axis=1),
  #     'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  # }

  predictions = {
      'classes': tf.round(tf.nn.sigmoid(logits)),
      'probabilities': tf.nn.sigmoid(logits, name='sigmoid_tensor'),
      'real_classes' : tf.round(tf.exp(logits)),
      'real_prob': tf.exp(logits)
  }


  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    # print('Prediction')
    # features = tf.Print(features, [tf.shape(features),features], 'feature', summarize=100)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  # cross_entropy = tf.losses.sparse_softmax_cross_entropy(
  #     logits=logits, labels=labels)
  # labels = tf.Print(labels, [labels, logits], 'input to loss function', summarize=1000)

  if model_method == 1:
      model_loss =  tf.cond(tf.equal(tf.shape(image_features)[1], 224),
                            true_fn = lambda : tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits),
                            false_fn= lambda : model.get_loss(logits, labels))
  elif model_method == 2:
      model_loss =  tf.cond(tf.equal(tf.shape(image_features)[1], 224),
                            true_fn = lambda : model.get_resnet_loss(logits, labels),
                            false_fn= lambda : model.get_loss(logits, labels))
  else:
      raise ValueError('invalid input for model method')

  # model_loss = tf.Print(model_loss, [tf.shape(logits)], 'logits shape')
  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(model_loss, name='model_loss')
  tf.summary.scalar('loss', model_loss)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = model_loss + l2_loss
  # loss = model_loss

  if mode == tf.estimator.ModeKeys.TRAIN:

    global_step = tf.train.get_or_create_global_step()
    # global_step = tf.Print(global_step, [global_step], 'global step')
    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # original model using monetum
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
    )

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.

      This function is used for fine tuning.

      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      # box_vars = []
      # for (g, v) in grad_vars:
      #     if 'resnet' not in v.name:
      #         print(v)
      #         box_vars.append((g, v))
      # loss = tf.Print(loss, [tf.reduce_mean(box_vars[0][0]), tf.reduce_mean(box_vars[1][0]), tf.reduce_mean(box_vars[2][0])], 'box var', summarize=12)
      # loss = tf.Print(loss, [loss], 'box var', summarize=12)

      # print(box_vars)
      # grad_vars = tf.Print(grad_vars,[grad_vars], summarize=1000)
      # tf.summary.scalar('gradients/AddN_2_0', tf.reduce_mean(grad_vars[0][0])) # 2048*80
      # tf.summary.scalar('resnet_model/dense/kernel_0', tf.reduce_mean(grad_vars[0][1])) #2048*80
      # tf.summary.scalar('gradients/AddN_1_0', tf.reduce_mean(grad_vars[1][0])) #80
      # tf.summary.scalar('resnet_model/dense/bias_0', tf.reduce_mean(grad_vars[1][1])) #80
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
    #     train_op= optimizer.minimize(
    #         loss=loss,
    #         global_step=tf.train.get_global_step())
  else:
    train_op = None

  def f1_metric_fn(labels=None, predictions=None):
    P, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    R, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    eps = 1e-5
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))

  zero = tf.constant(0, dtype=tf.int32)
  zero_float = tf.constant(0.0, dtype=tf.float32)
  # predictions['classes'] = tf.Print(predictions['classes'], [features], 'feature')
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.shape(labels), tf.shape(predictions['classes'])], 'label and prediction shape')
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.shape(logits), logits], 'logits', summarize=3000)
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.where(tf.not_equal(labels, zero)), tf.shape(labels)], 'labels', summarize=3000)
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.reduce_sum(tf.cast(tf.equal(labels, tf.cast(predictions['classes'], tf.int32)), tf.int32))], 'correct prediction', summarize=3000)
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.where(tf.not_equal(predictions['classes'], zero_float))],'prediction', summarize=3000)
  # predictions['classes'] = tf.Print(predictions['classes'], [tf.boolean_mask(predictions['probabilities'], tf.not_equal(predictions['classes'], zero_float))], 'predictions', summarize=3000)

  # threshold 0.5
  if model_method == 1:
      pred_class = predictions['classes']
      pred_probability = predictions['probabilities']
  elif model_method == 2:
      pred_class = predictions['real_classes']
      pred_probability = predictions['real_prob']
  else:
      raise ValueError('Invalid model method')

  accuracy = tf.metrics.accuracy(labels, pred_class)
  auc = tf.metrics.auc(labels, pred_probability)
  micro_f1 = tf.contrib.metrics.f1_score(labels, pred_probability)
  precision = tf.metrics.precision(tf.cast(labels, dtype=tf.int64), pred_class)
  recall = tf.metrics.recall(tf.cast(labels, dtype=tf.int64), pred_class)
  mAP = tf.py_func(eval.mAP_func, [tf.cast(labels, dtype=tf.int64), pred_probability], tf.float32)
  best_thres_accu = tf.py_func(eval.best_accuracy, [pred_probability, tf.cast(labels, dtype=tf.int64)], tf.float32)
  correlation =tf.contrib.metrics.streaming_pearson_correlation(predictions=pred_probability, labels=tf.cast(labels, tf.float32))

  metrics = {'accuracy': accuracy,
             'precision':precision,
             'recall': recall,
             'micro_f1': micro_f1,
             'auc': auc,
             'pearson_correlation': correlation}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy, name='train_accuracy')
  tf.identity(auc, name='train_auc')
  tf.identity(micro_f1, name='train_micro_f1')
  tf.identity(precision, name='train_precision')
  tf.identity(recall, name='train_recall')
  tf.identity(mAP, name='train_map')
  tf.identity(correlation, name='train_correlation')
  tf.identity(best_thres_accu, name='train_best_thres_accuracy')
  # for no streaming metric, print them out for each epoch

  # write to summary
  tf.summary.scalar('train_accuracy', accuracy[1])
  tf.summary.scalar('train_auc', auc[1])
  tf.summary.scalar('train_micro_f1', micro_f1[1])
  tf.summary.scalar('train_precision', precision[1])
  tf.summary.scalar('train_recall', recall[1])
  tf.summary.scalar('train_f1', (2*(precision[1]*recall[1])/(precision[1]+recall[1]+1e-5)))
  tf.summary.scalar('train_correlation', correlation[1])
  tf.summary.scalar('train_best_accuracy', best_thres_accu)
  tf.summary.scalar('train_map', mAP)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def resnet_main(
    flags_obj, model_function, input_fns, dataset_name, shape=None):
    model_helpers.apply_clean(flags.FLAGS)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # session_config = tf.ConfigProto(
    #     inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
    #     intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
    #     allow_soft_placement=True)
    #
    # distribution_strategy = distribution_utils.get_distribution_strategy(
    #     flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

    # run_config = tf.estimator.RunConfig(
    #     train_distribute=distribution_strategy, session_config=session_config)
    run_config = tf.estimator.RunConfig(model_dir=flags_obj.model_dir,
                                    save_checkpoints_steps=20,
                                    )


    if flags_obj.pretrained_model_checkpoint_path is not None:
        warm_start_settings = tf.estimator.WarmStartSettings(
            flags_obj.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
            'resnet_size': int(flags_obj.resnet_size),
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'resnet_version': int(flags_obj.resnet_version),
            'loss_scale': flags_core.get_loss_scale(flags_obj),
            'dtype': flags_core.get_tf_dtype(flags_obj),
            'fine_tune': flags_obj.fine_tune,
            'model_method': flags_obj.model_method
        })

    run_params = {
        'batch_size': flags_obj.batch_size,
        'dtype': flags_core.get_tf_dtype(flags_obj),
        'resnet_size': flags_obj.resnet_size,
        'resnet_version': flags_obj.resnet_version,
        'synthetic_data': flags_obj.use_synthetic_data,
        'train_epochs': flags_obj.train_epochs,
    }
    if flags_obj.use_synthetic_data:
        dataset_name = dataset_name + '-synthetic'

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('resnet', dataset_name, run_params,
                                    test_id=flags_obj.benchmark_test_id)

    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'model_loss': 'model_loss',
        'train_accuracy': 'train_accuracy',
        'train_map': 'train_map'
    }

    train_hooks = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                             every_n_iter=flags_obj.batch_size)
    if flags_obj.model_method == 1:
        input_function = input_fns[0]
    if flags_obj.model_method > 1:
        input_function = input_fns[0]
        input_fn_cond = input_fns[1]
        input_fn_marg = input_fns[2]

    def input_fn_train(num_epochs):
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=num_epochs,
            num_gpus=flags_core.get_num_gpus(flags_obj))

    def input_fn_eval():
        return input_function(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1)

    if flags_obj.model_method == 1:
        if flags_obj.eval_only or not flags_obj.train_epochs:
            schedule, n_loops = [0], 1
        else:
            # Compute the number of times to loop while training. All but the last
            # pass will train for `epochs_between_evals` epochs, while the last will
            # train for the number needed to reach `training_epochs`. For instance if
            #   train_epochs = 25 and epochs_between_evals = 10
            # schedule will be set to [10, 10, 5]. That is to say, the loop will:
            #   Train for 10 epochs and then evaluate.
            #   Train for another 10 epochs and then evaluate.
            #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
            n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
            schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
            schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.
        for cycle_index, num_train_epochs in enumerate(schedule):
            tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))
            if num_train_epochs:
                classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                                 hooks=[train_hooks], max_steps=flags_obj.max_train_steps)

            tf.logging.info('Starting to evaluate.')

            # flags_obj.max_train_steps is generally associated with testing and
            # profiling. As a result it is frequently called with synthetic data, which
            # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
            # eval (which is generally unimportant in those circumstances) to terminate.
            # Note that eval will run for max_train_steps each loop, regardless of the
            # global_step count.
            eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                               steps=flags_obj.max_train_steps)


    elif flags_obj.model_method > 1:
        # train 1 epoch of image, then 1 epoch of conditional probability, then 1 epoch of marginal probability
        # evaluate every 10 epochs
        for i in range(flags_obj.train_epochs):
            print('epoch', i)
            classifier.train(input_fn=lambda: input_fn_train(1),
                                 hooks=[train_hooks], max_steps=flags_obj.max_train_steps)
            # classifier.train(input_fn=lambda: input_fn_cond(flags_obj.cond_file, 5418, False, 1),
            #                      hooks=[train_hooks], max_steps=flags_obj.max_train_steps)
            # classifier.train(input_fn=lambda: input_fn_marg(flags_obj.marg_file, 80, True, 1),
            #                      hooks=[train_hooks], max_steps=flags_obj.max_train_steps)
            if i % 10 == 0:
                tf.logging.info('Starting to evaluate.')
                # eval_results = classifier.evaluate(input_fn=lambda: input_fn_eval(),
                #                                steps=flags_obj.max_train_steps)
                # benchmark_logger.log_evaluation_result(eval_results)
                # eval_results = classifier.evaluate(input_fn=lambda: input_fn_cond(flags_obj.cond_file, 5418, False, 1),
                #                     steps=flags_obj.max_train_steps)
                # benchmark_logger.log_evaluation_result(eval_results)
                # eval_results = classifier.evaluate(input_fn=lambda: input_fn_marg(flags_obj.marg_file, 80, True, 1),
                #                     steps=flags_obj.max_train_steps)
                # benchmark_logger.log_evaluation_result(eval_results)
                # if model_helpers.past_stop_threshold(
                #         flags_obj.stop_threshold, eval_results['accuracy']):
                #     break
    else:
        raise ValueError('Invalid model method parameter')




    if flags_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            shape, batch_size=flags_obj.batch_size)
        classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


def define_resnet_flags(resnet_size_choices=None):
  """Add flags and validators for ResNet."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name='resnet_version', short_name='rv', default='2',
      enum_values=['1', '2'],
      help=flags_core.help_wrap(
          'Version of ResNet. (1 or 2) See README.md for details.'))
  flags.DEFINE_bool(
      name='fine_tune', short_name='ft', default=False,
      help=flags_core.help_wrap(
          'If True do not train any parameters except for the final layer.'))
  flags.DEFINE_string(
      name='pretrained_model_checkpoint_path', short_name='pmcp', default=None,
      help=flags_core.help_wrap(
          'If not None initialize all the network except the final layer with '
          'these values'))
  flags.DEFINE_boolean(
      name="eval_only", default=False,
      help=flags_core.help_wrap('Skip training and only perform evaluation on '
                                'the latest checkpoint.'))
  # add box flags
  flags.DEFINE_string('cond_file', '/home/xiangl/workspace/synthesis/images/label_constraints2014/cond.tfrecord',
                      help=flags_core.help_wrap('conditional training file'))
  flags.DEFINE_string('marg_file', '/home/xiangl/workspace/synthesis/images/label_constraints2014/marg.tfrecord',
                      help=flags_core.help_wrap('marginal training file'))
  flags.DEFINE_float('cond_weight', 0.9, help=flags_core.help_wrap('weight on conditional prob loss'))
  flags.DEFINE_float('marg_weight', 0.1, help=flags_core.help_wrap('weight on marginal prob loss'))
  flags.DEFINE_float('reg_weight', 0.0001, help=flags_core.help_wrap('regularization parameter for universe'))
  flags.DEFINE_string('regularization_method', 'delta',
                      help=flags_core.help_wrap('method to regularizing the embedding, either delta or universe_edge'))

  flags.DEFINE_integer('box_batch_size', 5338,
                       help=flags_core.help_wrap('Batch size. Must divide evenly into the dataset sizes.'))
  flags.DEFINE_integer('embed_dim', 10,
                       help=flags_core.help_wrap('word embedding dimension'))

  # add whether train box or not
  flags.DEFINE_integer('model_method', 1, '1 represent for resnet only,'
                                          '2 represent for adding box, use the same feature vector,'
                                          '3 represent for adding box, use inclusion of boxes for conditional probability')

  choice_kwargs = dict(
      name='resnet_size', short_name='rs', default='50',
      help=flags_core.help_wrap('The size of the ResNet model to use.'))

  if resnet_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)
