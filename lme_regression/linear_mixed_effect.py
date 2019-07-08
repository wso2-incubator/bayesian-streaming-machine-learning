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

# We'll use the following directory to store files we download as well as our
# preprocessed dataset.
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle

from utils.data_generator import DataGenerator

tfd = tfp.distributions

seed = 7
tf.set_random_seed(seed)
np.random.seed(seed)


def get_data_dict(data):
    X_fixed = data[["X_%i" % i for i in range(d)]]
    X_random = data["Z"]
    X_cluster = data["cluster"]
    y = data["y"]

    return {
        "x_fixed": X_fixed,
        "x_random": X_random,
        "x_cluster": X_cluster,
        "y_in": y
    }


if __name__ == '__main__':

    d = 100
    num_clusters = c = 1000
    dg = DataGenerator(d, sigma_b=5.0, sigma_e=5.0)
    stat_interval = 10000

    n_iter = 10
    n_samples = 1
    batch_size = 1
    scale = 1e20
    model_type = "SVI"

    out_path = "stats"
    id = time.time()
    directory1 = "%s/%i/%s/" % (out_path, int(id), model_type)

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    sess = tf.InteractiveSession()

    x_cluster = tf.placeholder(tf.int32, [None], name="x_cluster")
    x_random = tf.placeholder(tf.float32, [None], name="x_random")
    x_fixed = tf.placeholder(tf.float32, [None, d], name="x_fixed")
    y_in = tf.placeholder(tf.float32, [None], name="y")

    step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.float32)
    step_increment = tf.assign(step, step + 1)
    step_reset = tf.assign(step, 0)

    loc_p_cluster = tf.Variable(tf.zeros([num_clusters]), trainable=False, name="loc_pcluster")
    scale_p_cluster = tf.Variable(tf.ones([num_clusters]), trainable=False, name="scale_pcluster")
    p_cluster = tfd.MultivariateNormalDiag(loc_p_cluster, scale_p_cluster)

    loc_pw = tf.Variable(tf.zeros([d]), trainable=False, name="loc_pw")
    scale_pw = tf.Variable(tf.ones([d]), trainable=False, name="scale_pw")
    p_w = tfd.Normal(loc_pw, scale_pw)

    q_cluster = tfd.MultivariateNormalDiag(tf.get_variable('loc_cluster', [num_clusters], dtype=tf.float32),
                                           tf.nn.softplus(tf.get_variable('scale_cluster', [num_clusters],
                                                                          dtype=tf.float32)))

    q_w = tfd.Normal(tf.get_variable('loc_w', [d], dtype=tf.float32),
                     tf.nn.softplus(tf.get_variable('scale_w', [d], dtype=tf.float32)))

    mu = tf.get_variable('intercept', [], tf.float32) + \
         tf.tensordot(x_fixed, q_w.sample(), axes=1) + \
         tf.gather(q_cluster.sample(), x_cluster) * x_random

    likelihood = tfd.Normal(mu, 1.0)

    p_log_lik = tf.reduce_mean([tf.reduce_sum(likelihood.log_prob(y_in)) for _ in range(n_samples)])
    kl_penalty = tf.reduce_sum([
        tf.reduce_sum(tfd.kl_divergence(q_cluster, p_cluster)),
        tf.reduce_sum(tfd.kl_divergence(q_w, p_w))
    ])

    if model_type == "SVI":
        elbo_loss = -p_log_lik + kl_penalty / scale
    elif model_type == "SSVB":
        elbo_loss = -p_log_lik + kl_penalty / step
    elif model_type == "BB-SVB":
        elbo_loss = -p_log_lik + kl_penalty
    else:
        raise Exception("Unknown model type")

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(elbo_loss)

    log_likelihood = tf.reduce_mean([likelihood.log_prob(y_in) for _ in range(100)])
    prediction = tf.reduce_mean(likelihood.sample(1000), axis=0)

    if model_type == "SSVB":
        update = loc_p_cluster.assign(q_cluster.loc), loc_pw.assign(q_w.loc)
    elif model_type == "BB-SVB":
        update = loc_p_cluster.assign(q_cluster.loc), scale_p_cluster.assign(q_cluster.stddev()), \
                 loc_pw.assign(q_w.loc), scale_pw.assign(q_w.scale)

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    arr_log_lik = []
    mse_arr = []
    mae_arr = []
    # default pattern used to generate data
    d_scale = [1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 1, 10, 10, 10, 10, 1, 1]
    # d_scale = [i / 10 for i in d_scale]

    _n = 0
    count_arr = []
    count = 0
    for ds in d_scale:

        train_clusters = (np.random.random_integers(10, 100, c) / ds).astype(np.int32)
        test_clusters = (np.random.random_integers(10, 100, c) / 10).astype(np.int32)

        print(np.sum(train_clusters), np.sum(test_clusters))

        train, test = dg.generate_train_test_splits(train_clusters, test_clusters)
        train, test = shuffle(train).astype(np.float32), shuffle(test).astype(np.float32)

        train = get_data_dict(train)
        test = get_data_dict(test)

        dg.add_drift(0.5, True)

        n, d = train['x_fixed'].shape
        _n += n
        for i in range(0, n, batch_size):
            sess.run(step_increment)
            count += 1
            for _ in range(n_iter):
                _, loss, kl = sess.run((train_op, elbo_loss, kl_penalty), feed_dict={
                    x_fixed: train['x_fixed'][i:batch_size + i],
                    x_random: train['x_random'][i:batch_size + i],
                    x_cluster: train['x_cluster'][i:batch_size + i],
                    y_in: train['y_in'][i:batch_size + i]
                })
                i += batch_size

            if model_type is not "SVI":
                sess.run(update)

            _log_likelihood = sess.run(log_likelihood, feed_dict={
                x_fixed: test['x_fixed'],
                x_random: test['x_random'],
                x_cluster: test['x_cluster'],
                y_in: test['y_in']
            })

            arr_log_lik.append(_log_likelihood)

            print(i, scale, loss, kl, _log_likelihood)

            if i % 50 == 0 or i == n - 1:
                pred = sess.run(prediction, feed_dict={
                    x_fixed: test['x_fixed'],
                    x_random: test['x_random'],
                    x_cluster: test['x_cluster']
                })

                mse, mae = np.sqrt(mean_squared_error(test['y_in'], pred)), mean_absolute_error(test['y_in'], pred)
                mse_arr.append(mse)
                mae_arr.append(mae)
                count_arr.append(count)
                print("==================================================%f, %f" %
                      (mse, mae))

        # pred, loss = sess.run((prediction, elbo_loss), feed_dict={
        #     x_fixed: test['x_fixed'],
        #     x_random: test['x_random'],
        #     x_cluster: test['x_cluster'],
        #     y_in: test['y_in']
        # })

        print("writing stats to directory : %s" % directory1)

        lik_df = pd.DataFrame({"likelihood": arr_log_lik})
        path = "%s/lik.csv" % (directory1)
        if not os.path.isfile(path):
            lik_df.to_csv(path)
        else:
            lik_df.to_csv(path, mode='a', header=False)

        err_df = pd.DataFrame({"count": count_arr, "mse": mse_arr, "mae": mae_arr})
        path = "%s/error.csv" % (directory1)
        if not os.path.isfile(path):
            err_df.to_csv(path)
        else:
            err_df.to_csv(path, mode='a', header=False)

        count_arr = []
        mse_arr = []
        mae_arr = []
        arr_log_lik = []

    print(loss, np.sqrt(mean_squared_error(test['y_in'], pred)),
          mean_absolute_error(test['y_in'], pred))

    from matplotlib import pyplot as plt

    plt.plot(test['y_in'], test['y_in'], ls=":")
    plt.scatter(test['y_in'], pred)
    plt.show()

    # plt.plot(range(0, _n), arr_log_lik)
    # plt.show()

    # plt.plot(range(0, _n, 50), mse_arr)
    # plt.plot(range(0, _n, 50), mae_arr)
    # plt.show()
