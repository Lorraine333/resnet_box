from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils
from box import MyBox
import tensorflow as tf

my_seed = 20180112
tf.set_random_seed(my_seed)

def model_fn(features, labels, mode, params):
    """
    Creates model_fn for Tensorflow estimator. This function takes features and input, and
    is responsible for the creation and processing of the Tensorflow graph for training, prediction and evaluation.

    Expected feature: {'image': image tensor }

    :param features: dictionary of input features
    :param labels: dictionary of ground truth labels
    :param mode: graph mode
    :param params: params to configure model
    :return: Estimator spec dependent on mode
    """
    learning_rate = params['learning_rate']
    """Initiate box embeddings"""
    mybox = softbox_model_fn(features, labels, mode, params)

    log_prob = mybox.log_prob

    if mode == tf.estimator.ModeKeys.PREDICT:
        return get_prediction_spec(log_prob)

    total_loss = mybox.get_loss(log_prob, labels, params)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return get_training_spec(learning_rate, total_loss)

    else:
        return get_eval_spec(log_prob, labels, total_loss)



def get_prediction_spec(log_cond_prob):
    """
    Creates estimator spec for prediction

    :param log_cond_prob: log prob for conditionals
    :param log_marg_prob: log prob for marginals
    :return: Estimator spec
    """
    predictions = {
        "probability": tf.exp(log_cond_prob)
    }
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)


def get_training_spec(learning_rate, loss):
    """
    Creates training estimator spec

    :param learning rate for optimizer
    :param joint_loss: loss op
    :return: Training estimator spec
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


def get_eval_spec(log_cond_prob, labels, loss):
    """
    Creates eval spec for tensorflow estimator
    :param log_cond_prob: log prob for conditionals
    :param log_marg_prob: log prob for marginals
    :param labels: ground truth labels for conditional and marginal
    :param loss: loss op
    :return: Eval estimator spec
    """
    eval_metric_ops = {
        "pearson_correlation":tf.contrib.metrics.streaming_pearson_correlation(
            predictions=tf.exp(log_cond_prob), labels=labels['prob'])
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops)

class softbox_model_fn(object):
    def __init__(self, features, labels, mode, params):
        self.label_size = params['label_size']
        self.embed_dim = params['embed_dim']
        self.prob_label = labels['prob']
        self.cond_weight = params['cond_weight']
        self.marg_weight = params['marg_weight']
        self.reg_weight = params['reg_weight']
        self.regularization_method = params['regularization_method']

        self.temperature = 1.0

        self.min_embed, self.delta_embed = init_word_embedding(self.label_size, self.embed_dim)
        self.log_prob = tf.cond(tf.equal(tf.shape(self.prob_label)[1], tf.constant(self.label_size)),
                     true_fn=lambda: self.softbox_marg(features, params, mode),
                     false_fn=lambda: self.softbox_cond(features, params, mode))
        self.log_prob = tf.Print(self.log_prob, [tf.equal(tf.shape(self.prob_label)[1], tf.constant(self.label_size))], '0 for marginal, 1 for conditional')


    def softbox_cond(self, features, params, mode):
        embed_dim = params['embed_dim']
        t1x = features['term1']
        t2x = features['term2']
        """cond log probability"""
        t1_box = get_word_embedding(t1x, self.min_embed, self.delta_embed)
        t2_box = get_word_embedding(t2x, self.min_embed, self.delta_embed)
        evaluation_logits = get_conditional_probability(t1_box, t2_box, embed_dim, self.temperature)
        return evaluation_logits

    def softbox_marg(self, features, params, mode):
        """marg log probability"""
        max_embed = self.min_embed + tf.exp(self.delta_embed)
        universe_min = tf.reduce_min(self.min_embed, axis=0, keepdims=True)
        universe_max = tf.reduce_max(max_embed, axis=0, keepdims=True)
        universe_volume = volume_calculation(MyBox(universe_min, universe_max), self.temperature)
        box_volume = volume_calculation(MyBox(self.min_embed, max_embed), self.temperature)
        predicted_marginal_logits = tf.log(box_volume) - tf.log(universe_volume)
        return predicted_marginal_logits

    def get_cond_loss(self, cond_log_prob):
        """get conditional probability loss"""
        cond_pos_loss = tf.multiply(cond_log_prob, self.prob_label)
        cond_neg_loss = tf.multiply(tf.log(1-tf.exp(cond_log_prob)+1e-10), 1-self.prob_label)
        cond_loss = -tf.reduce_mean(cond_pos_loss+ cond_neg_loss)
        cond_loss = self.cond_weight * cond_loss
        return cond_loss

    def get_marg_loss(self, marg_log_prob):
        """get marginal probability loss"""
        marg_pos_loss = tf.multiply(marg_log_prob, self.prob_label)
        marg_neg_loss = tf.multiply(tf.log(1-tf.exp(marg_log_prob)+1e-10), 1-self.prob_label)
        marg_loss = -tf.reduce_mean(marg_pos_loss+marg_neg_loss)
        marg_loss = self.marg_weight * marg_loss
        return marg_loss


    def get_loss(self, log_prob, labels, params):
        prob_loss = tf.cond(tf.equal(tf.shape(self.prob_label)[0], tf.constant(self.label_size)),
                        true_fn=lambda: self.get_marg_loss(log_prob),
                        false_fn=lambda: self.get_cond_loss(log_prob))

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


def volume_calculation(mybox, temperature):
    return tf.reduce_prod(tf.nn.softplus((mybox.max_embed - mybox.min_embed)/
                                         temperature)*temperature, axis=-1)

def init_embedding_scale():
    # softbox delta log init
    min_lower_scale, min_higher_scale = 1e-4, 0.9
    delta_lower_scale, delta_higher_scale = -1.0, -0.1
    # min_lower_scale, min_higher_scale = 1e-4, 0.9
    # delta_lower_scale, delta_higher_scale = -0.1, 0
    return min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale

def init_word_embedding(label_size, embed_dim):
    min_lower_scale, min_higher_scale, delta_lower_scale, delta_higher_scale = init_embedding_scale()
    min_embed = tf.Variable(
        tf.random_uniform([label_size, embed_dim],
                          min_lower_scale, min_higher_scale, seed=my_seed), trainable=True, name='word_embed')
    delta_embed = tf.Variable(
        tf.random_uniform([label_size, embed_dim],
                          delta_lower_scale, delta_higher_scale, seed=my_seed), trainable=True, name='delta_embed')
    return min_embed, delta_embed

def get_word_embedding(idx, min_embed, delta_embed):
    """read word embedding from embedding table, get unit cube embeddings"""
    min_embed = tf.nn.embedding_lookup(min_embed, idx)
    delta_embed = tf.nn.embedding_lookup(delta_embed, idx) # [batch_size, embed_size]
    max_embed = min_embed + tf.exp(delta_embed)
    t1_box = MyBox(min_embed, max_embed)
    return t1_box

def get_conditional_probability(t1_box, t2_box, embed_dim, temperature):
    _, meet_box, disjoint = utils.calc_join_and_meet(t1_box, t2_box)
    nested = utils.calc_nested(t1_box, t2_box, embed_dim)
    """get conditional probabilities"""
    overlap_volume = volume_calculation(meet_box, temperature)
    rhs_volume = volume_calculation(t1_box, temperature)
    conditional_logits = tf.log(overlap_volume+1e-10) - tf.log(rhs_volume+1e-10)
    return conditional_logits

