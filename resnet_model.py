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
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.resnet import softbox
from official.resnet.box import MyBox

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, with a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               final_size, resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size
    self.dtype = dtype
    self.pre_activation = resnet_version == 2

    self.embed_dim = 20
    self.cond_weight = 1.0
    self.marg_weight = 0.0
    self.reg_weight = 0.001
    self.temperature = 1.0
    self.regularization_method = 'delta' # or universe_edge


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)

  def get_cond_loss(self, cond_log_prob, labels):
      """get conditional probability loss"""
      cond_pos_loss = tf.multiply(cond_log_prob, labels)
      cond_neg_loss = tf.multiply(tf.log(1-tf.exp(cond_log_prob)+1e-10), 1-labels)
      cond_loss = -tf.reduce_mean(cond_pos_loss+ cond_neg_loss)
      cond_loss = self.cond_weight * cond_loss
      return cond_loss

  def get_marg_loss(self, marg_log_prob, labels):
      """get marginal probability loss"""
      marg_pos_loss = tf.multiply(marg_log_prob, labels)
      marg_neg_loss = tf.multiply(tf.log(1-tf.exp(marg_log_prob)+1e-10), 1-labels)
      marg_loss = -tf.reduce_mean(marg_pos_loss+marg_neg_loss)
      marg_loss = self.marg_weight * marg_loss
      return marg_loss

  def get_loss(self, log_prob, labels):
    labels = tf.cast(labels, tf.float32)
    prob_loss = tf.cond(tf.equal(tf.shape(labels)[1], tf.constant(self.num_classes)),
                        true_fn=lambda: self.get_marg_loss(log_prob, labels),
                        false_fn=lambda: self.get_cond_loss(log_prob, labels))

    """get regularization loss"""
    if self.regularization_method == 'universe_edge':
        max_embed = self.min_embed + tf.exp(self.delta_embed)
        universe_min = tf.reduce_min(self.min_embed, axis=0, keepdims=True)
        universe_max = tf.reduce_max(max_embed, axis=0, keepdims=True)
        regularization = tf.reduce_mean(
            tf.nn.softplus(universe_max - universe_min))
    elif self.regularization_method == 'delta':
        regularization = tf.reduce_mean(
            tf.square(tf.exp(self.delta_embed)))
    else:
        raise ValueError('Wrong regularization method')

    total_loss = prob_loss + self.reg_weight * regularization
    total_loss = tf.Print(total_loss, [prob_loss, self.reg_weight * regularization], 'loss')
    return total_loss


  def __call__(self, inputs, image_features, labels):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    self.min_embed, self.delta_embed = softbox.init_word_embedding(self.num_classes, self.embed_dim)

    return tf.cond(tf.equal(tf.shape(inputs['image'])[1], 224),
                   true_fn = lambda : self.resnet_call(image_features, labels),
                   false_fn=lambda : self.box_call(inputs, labels))

  def joint_model(self, inputs, image_features, labels):
    self.min_embed, self.delta_embed = softbox.init_word_embedding(self.num_classes, self.embed_dim)

    return tf.cond(tf.equal(tf.shape(inputs['image'])[1], 224),
                   true_fn = lambda : self.get_image_box_logits(image_features),
                   false_fn=lambda : self.box_call(inputs, labels))

  def box_call(self, inputs, labels):
      return tf.cond(tf.equal(tf.shape(labels)[1], tf.constant(self.num_classes)),
                     true_fn=lambda: self.marg_call(),
                     false_fn=lambda: self.cond_call(inputs))

  def cond_call(self, inputs):
      t1x = inputs['term1']
      t2x = inputs['term2']
      # t1x = tf.Print(t1x, [t1x, t2x], 't1x shape')
      """cond log probability"""
      t1_box = softbox.get_word_embedding(t1x, self.min_embed, self.delta_embed)
      t2_box = softbox.get_word_embedding(t2x, self.min_embed, self.delta_embed)
      evaluation_logits = softbox.get_conditional_probability(t1_box, t2_box, self.embed_dim, self.temperature)
      # evaluation_logits = tf.Print(evaluation_logits, [evaluation_logits], 'logit')
      return evaluation_logits

  def marg_call(self):
      """marg log probability"""
      max_embed = self.min_embed + tf.exp(self.delta_embed)
      universe_min = tf.reduce_min(self.min_embed, axis=0, keepdims=True)
      universe_max = tf.reduce_max(max_embed, axis=0, keepdims=True)
      universe_volume = softbox.volume_calculation(MyBox(universe_min, universe_max), self.temperature)
      box_volume = softbox.volume_calculation(MyBox(self.min_embed, max_embed), self.temperature)
      predicted_marginal_logits = tf.log(box_volume) - tf.log(universe_volume)
      predicted_marginal_logits = tf.expand_dims(predicted_marginal_logits, axis = 0)
      return predicted_marginal_logits

  def get_resnet_loss(self, logits, labels):
      """Returns an input function that returns a dataset with zeroes.

      This is useful in debugging input pipeline performance, as it removes all
      elements of file reading and image preprocessing.

      Args:
          resnet_logits: resnet_logits for resnet output. Shape [batch_size, num_classes]
          labels: one hot labels. Shape [batch_size, label_size]

      Returns:
          The loss function for resnet if model_method is 2
      """
      # image feature box volume
      labels = tf.cast(labels, tf.float32)
      # labels = tf.reshape(labels, [-1])
      # logits = tf.reshape(logits, [-1])
      # cond_pos_loss = tf.multiply(logits, labels)
      # cond_neg_loss = tf.multiply(tf.log(1-tf.exp(logits)+1e-10), 1-labels)
      # loss = -tf.reduce_mean(cond_pos_loss+ cond_neg_loss)
      # loss = -tf.reduce_mean(cond_pos_loss)
      loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
      return loss

  def get_image_box_logits(self, image_features):
      # labels = tf.Print(labels, [labels], 'label at resnet loss', summarize=100)
      # self.image_feature = self.resnet_feature_call(inputs, training) # feature layer output [batch_size, 2048]
      resnet_min_embed = tf.layers.dense(inputs=image_features, units=self.embed_dim) # [batch_size, 20]
      resnet_delta_embed = tf.layers.dense(inputs =image_features, units = self.embed_dim) # [batch_size, 20]

      resnet_min_embed = tf.tile(tf.expand_dims(resnet_min_embed, 1), [1, self.num_classes, 1])
      resnet_min_embed = tf.reshape(resnet_min_embed, [-1, self.embed_dim]) #[batch_size * num_class, 20]

      resnet_delta_embed = tf.tile(tf.expand_dims(resnet_delta_embed, 1), [1, self.num_classes, 1])
      resnet_delta_embed = tf.reshape(resnet_delta_embed, [-1, self.embed_dim]) #[batch_size * num_class, 20]

      # resnet_min_embed = tf.Print(resnet_min_embed, [tf.shape(resnet_min_embed)], 'after reshaping size', summarize=1000)

      resnet_max_embed = resnet_min_embed + resnet_delta_embed
      resnet_box = MyBox(resnet_min_embed, resnet_max_embed)

      # box label
      max_embed = self.min_embed + tf.exp(self.delta_embed) #[label_size, 20]
      # max_embed = tf.Print(max_embed, [tf.shape(image_features)], 'image feature shape')
      min_embed = tf.tile(tf.expand_dims(self.min_embed, 0), [tf.shape(image_features)[0], 1, 1])
      min_embed = tf.reshape(min_embed, [-1, self.embed_dim])

      max_embed = tf.tile(tf.expand_dims(max_embed, 0), [tf.shape(image_features)[0], 1, 1])
      max_embed = tf.reshape(max_embed, [-1, self.embed_dim])
      # max_embed = tf.Print(max_embed, [tf.shape(max_embed), tf.shape(resnet_delta_embed), max_embed], 'box_label_shape after tile')
      label_box = MyBox(min_embed, max_embed)

      cond_log_prob = softbox.get_conditional_probability(resnet_box, label_box, self.embed_dim, self.temperature) #[batch_size * num_class]
      cond_log_prob = tf.reshape(cond_log_prob, [-1, self.num_classes])
      return cond_log_prob

  def resnet_call(self, inputs, training):
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      return inputs

  def resnet_feature_call(self, inputs, training):
    inputs = inputs['image']
    inputs = tf.cast(inputs, dtype=tf.float32)

    # inputs = tf.Print(inputs, [tf.shape(inputs)], 'shape', summarize=10)
    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
      # inputs = tf.Print(inputs, [inputs], 'input after channel first')
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
          strides=self.conv_stride, data_format=self.data_format)
      # inputs = tf.Print(inputs, [inputs], 'input after conv2d', summarize=100)
      inputs = tf.identity(inputs, 'initial_conv')

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if self.resnet_version == 1:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)
      # inputs = tf.Print(inputs, [inputs], 'input after resnet version', summarize=100)

      if self.first_pool_size:
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
      # inputs = tf.Print(inputs, [inputs], 'input after pool size', summarize=100)

      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format)
      # inputs = tf.Print(inputs, [inputs], 'input after block ', summarize=100)

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.reshape(inputs, [-1, self.final_size])
      self.image_feature = inputs
      return inputs
