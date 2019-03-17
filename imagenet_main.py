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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.resnet import easier_preprocessing
from official.resnet import resnet_model
from official.resnet import resnet_run_loop

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
# _NUM_CLASSES = 1001
_NUM_CLASSES = 80
_NUM_COND_PROB = 5418

# _NUM_IMAGES = {
#     'train': 1281167,
#     'validation': 50000,
# }
_NUM_IMAGES = {
    # 'train': 206,
    # 'validation': 8,
    # 'train': 1,
    # 'validation': 1,
    # 2014 coco
    'train': 82081, # 702 without annotation, 82783 examples, 81*1024=82944
    'validation': 40137, #364 without anntations, 40501 examples, 413*128=40192
}

_NUM_TRAIN_FILES = 1024
_NUM_VALID_FILES = 128
# _NUM_TRAIN_FILES = 1
# _NUM_VALID_FILES = 1
_SHUFFLE_BUFFER = 10000

# DATASET_NAME = 'ImageNet'
DATASET_NAME = 'MSCoCo2014'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train2014-%05d-of-%05d' % (i, _NUM_TRAIN_FILES))
        # os.path.join(data_dir, 'tinytinytiny_valid-%05d-of-%05d' % (i, _NUM_VALID_FILES))
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'val2014-%05d-of-%05d' % (i, _NUM_VALID_FILES))
        # os.path.join(data_dir, 'tinytinytiny_valid-%05d-of-%05d' % (i, _NUM_VALID_FILES))
        for i in range(_NUM_VALID_FILES)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/format': tf.FixedLenFeature((), dtype=tf.string,
                                         default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  # sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  # feature_map.update(
  #     {k: sparse_float32 for k in ['image/object/bbox/xmin',
  #                                  'image/object/bbox/ymin',
  #                                  'image/object/bbox/xmax',
  #                                  'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  # label = tf.cast(features['image/class/label'], dtype=tf.int32)
  label = tf.decode_raw(features['image/class/label'], tf.int32)

  # xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  # ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  # xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  # ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  # bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  # bbox = tf.expand_dims(bbox, 0)
  # bbox = tf.transpose(bbox, [0, 2, 1])

  # return features['image/encoded'], label, bbox
  image = features['image/encoded']
  # image = tf.image.decode_image(image, _NUM_CHANNELS)
  image = tf.image.decode_image(
      tf.reshape(image, shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image, label


def parse_record(raw_record, is_training):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  # image_buffer, label, bbox = _parse_example_proto(raw_record)
  #
  # image = imagenet_preprocessing.preprocess_image(
  #     image_buffer=image_buffer,
  #     bbox=bbox,
  #     output_height=_DEFAULT_IMAGE_SIZE,
  #     output_width=_DEFAULT_IMAGE_SIZE,
  #     num_channels=_NUM_CHANNELS,
  #     is_training=is_training)


  image_buffer, label = _parse_example_proto(raw_record)

  image = easier_preprocessing.preprocess_image(
      image=image_buffer,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)
  # image = tf.Print(image, [image, label], 'input', summarize=100)
  features_image = {'image': image,
                    'idx': tf.zeros_like(label),
                    'term1': tf.zeros_like(label),
                    'term2': tf.zeros_like(label)}
  label_dict = {'prob': label}
  return features_image, label_dict


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.contrib.data.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))
  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )

def box_cond_input_fn(filename, batch_size, shuffle, epoch):

    def parse_cond_feature(raw_record):
        """"
        return a tuple of feature and label
        feature is a dictionay of three tensors, first two tensors with shape [example_size]
        third one with shape [label_size]
        label is a tensor with shape [example_size]
        """""
        # Define features
        feature_map={
            'u': tf.FixedLenFeature([5418], dtype=tf.int64),
            'v': tf.FixedLenFeature([5418], dtype=tf.int64),
            'prob': tf.FixedLenFeature([5418], dtype=tf.float32)}

        features = tf.parse_single_example(raw_record, feature_map)
        return_features = {
            'image': tf.zeros([32, 32, 3], dtype=tf.float32),
            'idx': tf.zeros_like(features['u']),
            'term1': features['u'],
            'term2': features['v']}
        labels = {'prob': features['prob']}
        return return_features, labels

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_cond_feature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(batch_size * 10)
    # return dataset

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def box_marg_input_fn(filename, batch_size, shuffle, epoch):
    def parse_marg_feature(raw_record):
        """"
        return a tuple of feature and label
        feature is a dictionay of three tensors, first two tensors with shape [example_size]
        third one with shape [label_size]
        label is a tensor with shape [example_size]
        """""
        # Define features

        feature_map={
            'marg_idx' : tf.FixedLenFeature([80], dtype=tf.int64),
            'marg_prob': tf.FixedLenFeature([80], dtype=tf.float32)}
        features = tf.parse_single_example(raw_record, feature_map)

        return_features = {
            'image': tf.zeros([32, 32, 3], dtype=tf.float32),
            'idx': features['marg_idx'],
            'term1': tf.zeros_like(features['marg_idx']),
            'term2': tf.zeros_like(features['marg_idx'])}
        labels = {'prob': features['marg_prob']}

        return return_features, labels

    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_marg_feature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(batch_size * 10)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
    # return dataset

def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)

###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=warmup, base_lr=base_lr)

  def only_resnet_fn(input_var):
      if 'resnet' in input_var:
          return True
      else:
          return False

  resnet_fn = resnet_run_loop.resnet_model_fn(
      input_features=features,
      input_labels=labels,
      mode=mode,
      model_class=ImagenetModel,
      resnet_size=params['resnet_size'],
      weight_decay=1e-3,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=only_resnet_fn,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
      model_method = params['model_method']
  )

  return resnet_fn



def define_imagenet_flags():
  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'])
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(train_epochs=100)


def run_imagenet(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_fns = []
  if flags_obj.use_synthetic_data:
      input_fns.append(get_synth_input_fn())
# means using boxes
  if flags_obj.model_method == 1:
      input_fns.append(input_fn)
  elif flags_obj.model_method > 1:
      input_fns.append(input_fn)
      input_fns.append(box_cond_input_fn)
      input_fns.append(box_marg_input_fn)

  else:
      raise ValueError('invalid input for model method')

  resnet_run_loop.resnet_main(
      flags_obj, imagenet_model_fn, input_fns, DATASET_NAME,
      shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_imagenet(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_imagenet_flags()
  absl_app.run(main)
