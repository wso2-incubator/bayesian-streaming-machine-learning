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
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle

from utils.data_generator_linear import DataGenerator

tfd = tfp.distributions

seed = 7
tf.set_random_seed(seed)
np.random.seed(seed)


def get_data_dict(data):
    X_fixed = data[["X_%i" % i for i in range(d)]]
    X_random = data["Z"]
    X_cluster = data["cluster"]
    # num_clusters = np.unique(X_cluster).shape[0]
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

    n_iter = 2
    batch_size = 1

    id = time.time()
    directory1 = "%s/%i/%s/" % ("other_iter1", int(id), "PA")
    directory2 = "%s/%i/%s/" % ("other_iter1", int(id), "SGD")

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    if not os.path.exists(directory2):
        os.makedirs(directory2)

    pa_mse_arr = []
    pa_mae_arr = []
    sgd_mse_arr = []
    sgd_mae_arr = []

    d_scale = [1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 1, 10, 10, 10, 10, 1, 1]
    # d_scale = [i / 10 for i in d_scale]

    pa = PassiveAggressiveRegressor(n_iter=n_iter)
    sgd = SGDRegressor(max_iter=n_iter, learning_rate="constant", alpha=0.001)

    _n = 0
    count_arr = []
    count = 0
    for ds in d_scale:

        train_clusters = (np.random.random_integers(10, 100, c) / ds).astype(np.int32)
        test_clusters = (np.random.random_integers(10, 100, c) / 10).astype(np.int32)

        print(np.sum(train_clusters), np.sum(test_clusters))

        train, test = dg.generate_train_test_splits(train_clusters, test_clusters)
        train, test = shuffle(train).astype(np.float32), shuffle(test).astype(np.float32)

        train_x, train_y = train.iloc[:, 1:], train.iloc[:, 0]
        test_x, test_y = test.iloc[:, 1:], test.iloc[:, 0]

        dg.add_drift(0.5, True)

        n, d = train_x.shape
        _n += n
        for i in range(0, n, batch_size):
            count += 1
            pa.partial_fit(train_x[i:i + batch_size], train_y[i:i + batch_size])
            sgd.partial_fit(train_x[i:i + batch_size], train_y[i:i + batch_size])

            if i % 50 == 0 or i == n - 1:
                pred1 = pa.predict(test_x)
                pred2 = sgd.predict(test_x)

                mse1, mae1 = np.sqrt(mean_squared_error(test_y, pred1)), mean_absolute_error(test_y, pred1)
                mse2, mae2 = np.sqrt(mean_squared_error(test_y, pred2)), mean_absolute_error(test_y, pred2)

                pa_mse_arr.append(mse1)
                pa_mae_arr.append(mae1)
                sgd_mse_arr.append(mse2)
                sgd_mae_arr.append(mae2)
                count_arr.append(count)

                print("pa %f %f, sgd %f %f" % (mse1, mae1, mse2, mae2))
        print("writing stats to directory : %s" % directory1)

        err_df = pd.DataFrame({"count": count_arr, "mse": pa_mse_arr, "mae": pa_mae_arr})
        path = "%s/error.csv" % (directory1)
        if not os.path.isfile(path):
            err_df.to_csv(path)
        else:
            err_df.to_csv(path, mode='a', header=False)

        err_df = pd.DataFrame({"count": count_arr, "mse": sgd_mse_arr, "mae": sgd_mae_arr})
        path = "%s/error.csv" % (directory2)
        if not os.path.isfile(path):
            err_df.to_csv(path)
        else:
            err_df.to_csv(path, mode='a', header=False)

        count_arr = []
        pa_mse_arr = []
        pa_mae_arr = []
        sgd_mse_arr = []
        sgd_mae_arr = []
