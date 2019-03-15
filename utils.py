from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from random import *
from official.resnet.box import MyBox

random_seed = 20180112


def calc_join_and_meet(t1_box, t2_box):
    """
    # two boxes are t1_box, t2_box
    # the corresponding embeddings are t1_min_embed, t1_max_embed, t2_min_embed, t2_max_embed
    Returns:
        join box, min box, and disjoint condition:
    """
    # join is min value of (a, c), max value of (b, d)
    t1_min_embed = t1_box.min_embed
    t1_max_embed = t1_box.max_embed
    t2_min_embed = t2_box.min_embed
    t2_max_embed = t2_box.max_embed
    join_min = tf.minimum(t1_min_embed, t2_min_embed)
    join_max = tf.maximum(t1_max_embed, t2_max_embed)
    # find meet is calculate the max value of (a,c), min value of (b,d)
    meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
    meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
    # The overlap cube's max value have to be bigger than min value in every dimension to form a valid cube
    # if it's not, then two concepts are disjoint, return none
    cond = tf.cast(tf.less_equal(meet_max, meet_min), tf.float32)  # batchsize * embed_size
    cond = tf.cast(tf.reduce_sum(cond, axis=1), tf.bool)  # batchsize. If disjoint, cond > 0; else, cond = 0
    meet_box = MyBox(meet_min, meet_max)
    join_box = MyBox(join_min, join_max)
    return join_box, meet_box, cond

def calc_nested(t1_box, t2_box, embed_size):
    t1_min_embed = t1_box.min_embed
    t1_max_embed = t1_box.max_embed
    t2_min_embed = t2_box.min_embed
    t2_max_embed = t2_box.max_embed
    meet_min = tf.maximum(t1_min_embed, t2_min_embed)  # batchsize * embed_size
    meet_max = tf.minimum(t1_max_embed, t2_max_embed)  # batchsize * embed_size
    cond1 = tf.cast(tf.equal(meet_max, t1_max_embed), tf.float32)
    cond2 = tf.cast(tf.equal(meet_min, t1_min_embed), tf.float32)
    cond3 = tf.cast(tf.equal(meet_max, t2_max_embed), tf.float32)
    cond4 = tf.cast(tf.equal(meet_min, t2_min_embed), tf.float32)
    cond5 = tf.equal(
        tf.reduce_sum(cond1, axis=1) + tf.reduce_sum(cond2, axis=1), embed_size*2)
    cond6 = tf.equal(
        tf.reduce_sum(cond3, axis=1) + tf.reduce_sum(cond4, axis=1), embed_size*2)
    return tf.logical_or(cond5, cond6)

class DataSet(object):
    def __init__(self, input_tuples):
        """Construct a DataSet"""
        self._num_examples = len(input_tuples)
        self._tuples = input_tuples
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # seed(random_seed)
            shuffle(self._tuples)
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        next_batch = self._tuples[start:end]
        batch_idx = [i for i in next_batch]
        t1_idx = [i[0] for i in batch_idx]
        t2_idx = [i[1] for i in batch_idx]
        s = [i[2] for i in batch_idx]
        return t1_idx, t2_idx, s


def read_data_sets(FLAGS, dtype=tf.float32):
    class DataSets(object):
        pass

    data_sets = DataSets()
    TRAIN_FILE = FLAGS.cond_file
    MARG_FILE = FLAGS.marg_file

    train_data = get_data(TRAIN_FILE)
    data_sets.train = DataSet(train_data)

    # if the loss term minimize marginal prob as well, then read in the marginal file
    marginal_prob = get_count(MARG_FILE)
    data_sets.marginal_prob = marginal_prob

    return data_sets

def get_data(filename):
    """Read data: idx1 \t idx2 \t score """
    f = open(filename, 'r')
    lines = f.readlines()
    examples = []
    for i in lines:
        i = i.strip()
        if (len(i) > 0):
            i = i.split('\t')
            e = (i[0], i[1], float(i[2]))
            examples.append(e)
    seed(random_seed)
    shuffle(examples)
    f.close()
    print(examples[:3])
    print('read data from', filename, 'of length', len(examples))
    return examples


def get_count(filename):
    """Read in marginal prob, to form a matrix of size: vocab * 1"""
    count = []
    with open(filename) as inputfile:
        lines = inputfile.read().splitlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            count.append(line)
    return np.asarray(count, dtype=np.float32)
