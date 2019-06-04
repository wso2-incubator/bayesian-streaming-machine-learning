"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import mlab, pyplot as plt

# from matplotlib.backends import backend_agg
# from tensorflow_probability import edward2 as ed
# import pandas as pd


# Dependency imports

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

tfd = tfp.distributions

n_samples = 1
num_examples = 1000
num_dim = 4
batch_size = 1


def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
    """Generates synthetic data for binary classification.
    Args:
      num_examples: The number of samples to generate (scalar Python `int`).
      input_size: The input space dimension (scalar Python `int`).
      weights_prior_stddev: The prior standard deviation of the weight
        vector. (scalar Python `float`).
    Returns:
      random_weights: Sampled weights as a Numpy `array` of shape
        `[input_size]`.
      random_bias: Sampled bias as a scalar Python `float`.
      design_matrix: Points sampled uniformly from the cube `[-1,
         1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
         input_size)`.
      labels: Labels sampled from the logistic model `p(label=1) =
        logistic(dot(inputs, random_weights) + random_bias)`, as a Numpy
        `int32` `array` of shape `(num_examples, 1)`.
    """
    random_weights = weights_prior_stddev * np.random.randn(input_size)
    random_bias = np.random.randn()
    design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
    logits = np.reshape(
        np.dot(design_matrix, random_weights) + random_bias,
        (-1, 1))
    p_labels = 1. / (1 + np.exp(-logits))
    # labels = np.int32(p_labels > np.random.rand(num_examples, 1))
    labels = np.int32(p_labels > np.random.normal(0.5, 0.2, num_examples).reshape(-1, 1))

    return random_weights, random_bias, np.float32(design_matrix), labels


def logistic_regression(inputs):
    """
    Logistic regression inference network
    :param inputs:
    :return:
    """
    d = inputs.shape[-1]

    qw = tfd.Normal(loc=tf.get_variable("loc_qw", [d]), scale=tf.nn.softplus(tf.get_variable("scale_qw", [d])))
    y = tfd.Bernoulli(logits=tf.tensordot(inputs, qw.sample(), axes=1))

    return y, qw


def init_model(num_dim, scale=1):
    # feed data mini-batches into the model
    inputs = tf.placeholder(tf.float32, [None, num_dim], "input_X")
    labels = tf.placeholder(tf.int32, [None], "input_y")

    # building the inference network
    likelihood, qw = logistic_regression(inputs)

    # defining the priors
    loc_pw = tf.placeholder(tf.float32, [num_dim], "loc_pw")
    scale_pw = tf.placeholder(tf.float32, [num_dim], "scale_pw")
    pw = tfd.Normal(loc=loc_pw, scale=scale_pw)

    # loss computation
    p_log_lik = [0.0] * n_samples  # likelihood
    for s in range(n_samples):  # mc-integration of the expectation of log likelihood
        p_log_lik[s] = tf.reduce_sum(likelihood.log_prob(labels))

    p_log_lik = -tf.reduce_mean(p_log_lik) * scale

    # computing the kl-penalty between q(w) and p(w)
    kl_penalty = tf.reduce_sum([tfd.kl_divergence(qw, pw)])

    elbo_loss = p_log_lik + kl_penalty

    opt = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = opt.minimize(elbo_loss)

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    probas = tf.reduce_mean(tf.cast(likelihood.sample(10000), dtype=tf.float32), axis=0)
    predictions = tf.cast(probas > 0.5, dtype=tf.int32)
    accuracy, accuracy_update_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions)

    return train_op, elbo_loss, kl_penalty, predictions, accuracy_update_op, accuracy, inputs, labels, loc_pw, scale_pw, qw


if __name__ == '__main__':

    scale = num_examples / batch_size
    stat_interval = 200
    w_true, b_true, x, y = toy_logistic_data(num_examples, num_dim)
    y = y.flatten()

    num_dim = num_dim + 1
    x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

    print("================== Start SVI - online ========================")

    train_op, elbo_loss, kl_penalty, predictions, \
    accuracy_update_op, accuracy, \
    inputs, labels, loc_pw, scale_pw, qw = init_model(num_dim, scale)

    # starting the session
    sess = tf.InteractiveSession()

    # initializing the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # SVI - online
    svi_loc_arr = []
    svi_scale_arr = []
    for i in range(0, num_examples, batch_size):
        _x, _y = x[i:i + batch_size], y[i:i + batch_size]
        _, accuracy_score = sess.run([accuracy_update_op, accuracy], feed_dict={inputs: _x, labels: _y})

        loc_p, scale_p = np.zeros((num_dim)), np.ones((num_dim))

        _, loss_value, kl = sess.run([train_op, elbo_loss, kl_penalty],
                                     feed_dict={inputs: _x, labels: _y, loc_pw: loc_p, scale_pw: scale_p})
        print(i, loss_value, kl, accuracy_score)

        if i % stat_interval == 0 or i == num_examples - 1:
            svi_loc_arr.append(qw.loc.eval())
            svi_scale_arr.append(qw.scale.eval())

    # Visualize some draws from the weights posterior.
    # w_draw = qw.sample()
    # candidate_w_bs = []
    # for _ in range(50):
    #     w = sess.run(w_draw)
    #     candidate_w_bs.append((w[:2], w[2]))
    # visualize_decision(x, y, (w_true, b_true),
    #                    candidate_w_bs, "BB-SVI"
    #                    )

    svi_loc = qw.loc.eval()
    svi_scale = qw.scale.eval()

    sess.close()
    tf.reset_default_graph()

    print("================== Start SDA-Bayes ========================")
    sda_loc_arr = []
    sda_scale_arr = []
    train_op, elbo_loss, kl_penalty, predictions, \
    accuracy_update_op, accuracy, \
    inputs, labels, loc_pw, scale_pw, qw = init_model(num_dim)

    # starting the session
    sess = tf.InteractiveSession()

    # initializing the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    loc_p, scale_p = np.zeros((num_dim)), np.ones((num_dim))

    for i in range(0, num_examples, batch_size):
        _x, _y = x[i:i + batch_size], y[i:i + batch_size]
        # _, accuracy_score = sess.run([accuracy_update_op, accuracy], feed_dict={inputs: _x, labels: _y})

        _, loss_value, kl = sess.run([train_op, elbo_loss, kl_penalty],
                                     feed_dict={inputs: _x, labels: _y, loc_pw: loc_p, scale_pw: scale_p})
        print(i, loss_value, kl)

        loc_p, scale_p = qw.loc.eval(), qw.scale.eval()
        sess.run(tf.global_variables_initializer())

        if i % stat_interval == 0 or i == num_examples - 1:
            sda_loc_arr.append(qw.loc.eval())
            sda_scale_arr.append(qw.scale.eval())

    svb_loc = qw.loc.eval()
    svb_scale = qw.scale.eval()

    sess.close()
    tf.reset_default_graph()

    w = np.append(w_true, b_true)
    fig, axes = plt.subplots(len(svi_loc_arr), num_dim)

    for iter in range(len(svi_loc_arr)):
        svi_loc = svi_loc_arr[iter]
        svi_scale = svi_scale_arr[iter]
        svb_loc = sda_loc_arr[iter]
        svb_scale = sda_scale_arr[iter]

        for i in range(num_dim):
            ax = axes[iter][i]
            if iter == 0:
                ax.set_title('$d = %d$' % (i + 1), size=20)

            if i == 0:
                ax.set_ylabel("$n = %d$" % (stat_interval * iter), fontdict={'size' : 20})

            # make xaxis invisibel
            ax.xaxis.set_visible(False)
            # make spines (the box) invisible
            plt.setp(ax.spines.values(), visible=False)
            # remove ticks and labels for the left axis
            ax.tick_params(left=False, labelleft=False)
            # remove background patch (only needed for non-white background)
            ax.patch.set_visible(False)

            mu_max, mu_min = max(svi_loc[i], svb_loc[i]), min(svi_loc[i], svb_loc[i])
            sigma = max(svi_scale[i], svb_scale[i])
            x = np.linspace(mu_min - 3 * sigma, mu_max + 3 * sigma, 1000)

            mu, sigma = svi_loc[i], svi_scale[i]
            ax.plot(x, mlab.normpdf(x, mu, sigma), label="BB-SVI", c="#ff7f0e")

            # ax = axes[1][i]
            mu, sigma = svb_loc[i], svb_scale[i]
            ax.plot(x, mlab.normpdf(x, mu, sigma), label="SVB", c="#9467bd", )

            ax.axvline(w[i], color='black', ls=":", label="true coefficient")
            ax.set_xticks([])
            ax.set_yticks([])

    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=3, prop={'size': 15})

    plt.show()
