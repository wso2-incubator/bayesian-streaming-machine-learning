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

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.sparse.csr import csr_matrix
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import MinMaxScaler

from classification.benchmark.arow import M_AROW, AROW
# Dependency imports
from classification.model.bayesian_classification import BayesianClassifier

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

tfd = tfp.distributions


class DummyModel:
    def fit(self, X, y, iter=None):
        return 0, 0

    def partial_fit(self, X, y, c):
        return 0, 0

    def predict(self, X):
        return [0] * X.shape[0]

    def log_predictive_likelihood(self, X, y):
        return [0]


def category_to_int(df, columns):
    """
    covert categories to integer codes
    :param df: pandas data-frame
    :param columns: columns to process. can be a string or a list of strings
    :return:
    """
    for col in columns:
        df[col] = df[col].astype('category')

    df[columns] = df[columns].apply(lambda x: x.cat.codes)

    return df


def load_usenet_dataset():
    """
    :return:
    """
    pt_data = pd.read_csv("../../../data/drift/usenet_recurrent3.3.data")

    data, target = pt_data.ix[:, :-1].astype('category').apply(lambda x: x.cat.codes), \
                   pt_data.ix[:, -1].astype('category').cat.codes.values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return data, target


def load_kddcup_dataset(limit=None):
    """
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("../../../data/drift/kddcup.data_10_percent", names=[str(i) for i in range(42)], header=None)

    data, target = category_to_int(pt_data.ix[:, :-1], ['1', '2', '3']), \
                   pt_data.ix[:, -1].astype('category').cat.codes.values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def load_generated_dataset(limit=None):
    """
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("../../../data/generated/small_data/data.csv")

    data, target = pt_data.ix[:, :-1].values, pt_data.ix[:, -1].values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def load_airlines_dataset(limit=None):
    """
    :param random_state:
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("%s/airlines.csv" % airline_dir)

    data, target = category_to_int(pt_data.ix[:, 1:-1], ["Unique Carrier", "Origin", "Destination"])[[
        "Month",
        "Day of Month",
        "Day of Week",
        "Actual Elapsed Time",
        "CRS Departure Time",
        "CRS Arrival Time",
        # "Unique Carrier",
        # "Flight Number",
        # "Origin",
        # "Destination",
        "Distance"
    ]], \
                   (pt_data.ix[:, -1] > 0).astype(np.int32).values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # train_X, test_X, train_y, test_y = train_test_split(data, target.astype(np.int32))
    if limit is not None:
        data, target = data[:limit], target[:limit]

    # data, target = shuffle(data, target.astype(np.int32), random_state=random_state)

    return data, target


def load_sea_dataset(limit=None):
    """
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("../../../data/drift/sea.data", names=[str(i) for i in range(4)], header=None)

    data, target = pt_data.ix[:, :-1].values, \
                   pt_data.ix[:, -1].values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def load_electricity_dataset(limit=None):
    """

    :param limit:
    :return:
    """
    pt_data = pd.read_csv("../../../data/drift/elecNormNew.csv")

    data, target = pt_data.ix[:, :-1].values, \
                   pt_data.ix[:, -1].astype('category').cat.codes.values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def load_forest_cover_type(limit=None):
    """
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("../../../data/drift/covtypeNorm.csv")

    data, target = pt_data.ix[:, :-1].values, \
                   pt_data.ix[:, -1].values - 1

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def load_pocker_hand(limit=None):
    """
    :param random_state:
    :param limit:
    :return:
    """
    pt_data = pd.read_csv("%s/poker-lsn.csv" % poker_dir)

    data, target = pt_data.ix[:, :-1].values, \
                   pt_data.ix[:, -1].values

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    if limit is not None:
        data, target = data[:limit], target[:limit]

    return data, target


def get_minibatch(X, y, current_idx, batch_size=1):
    X, y = X[current_idx:current_idx + batch_size], y[current_idx:current_idx + batch_size]
    return X.toarray() if isinstance(X, csr_matrix) else X, y


def run_exp(outpath, model_list, datasets, n_iter, alphas=None, proportion=None):
    n_samples = 1
    batch_size = 1
    stat_rate = 10000

    id = time.time()
    directory1 = "%s/%i" % (outpath, int(id))

    if not os.path.exists(directory1):
        os.makedirs(directory1)

    for dataset in datasets:
        directory2 = "%s/%s" % (directory1, dataset)
        if not os.path.exists(directory2):
            os.makedirs(directory2)

        if dataset == "kdd":
            data = load_kddcup_dataset()
        elif dataset == "airlines":
            data = load_airlines_dataset()
        elif dataset == "poker":
            data = load_pocker_hand()
        elif dataset == "usenet":
            data = load_usenet_dataset()
        elif dataset == "covertype":
            data = load_forest_cover_type()
        elif dataset == "electricity":
            data = load_electricity_dataset()
        elif dataset == "sea":
            data = load_sea_dataset()
        elif dataset == "generated":
            data = load_generated_dataset()
        else:
            raise Exception("Dataset %s is not found" % dataset)

        train_data, train_target = data

        # TODO remove
        # train_data, train_target = train_data[:10000], train_target[:10000]

        n, d = train_data.shape
        c = np.unique(train_target).shape[0]
        scale = (n) / float(batch_size)

        if proportion is not None:
            n = int(n * proportion)
            train_data, train_target = train_data[:n], train_target[:n]

        lr = datasets[dataset]

        print("Data set summery : number of samples %i, number of features %i, number of classes %i" % (n, d, c))

        # create models
        models = {}
        bmodels = []
        for model in model_list:
            if model == "AROW":
                if c > 2:
                    # m_model = sol.SOL('arow', c)
                    m_model = M_AROW(c, d, n_iter=n_iter)  # TODO verify the arrow implementation
                    models[model] = m_model
                else:
                    m_model = AROW(d, n_iter=n_iter)
                    models[model] = m_model
            elif model == "SGD":
                m_model = SGDClassifier(max_iter=n_iter, tol=1)
                models[model] = m_model

            elif model == "PA":
                m_model = PassiveAggressiveClassifier(max_iter=n_iter, tol=1)
                models[model] = m_model

            elif model == "BB-SVB":
                m_model = BayesianClassifier("BB-SVB", n_samples, d, c, learning_rate=lr)
                bmodels.append("BB-SVB")
                models[model] = m_model

            elif model == "SSVB":
                m_model = BayesianClassifier("SSVB", n_samples, d, c, learning_rate=lr)
                bmodels.append("SSVB")
                models[model] = m_model

            elif model == "SVI":
                m_model = BayesianClassifier("SVI", n_samples, d, c, learning_rate=lr, scale=scale)
                bmodels.append("SVI")
                models[model] = m_model

            elif model == "PVI":
                if alphas is None or len(alphas) < 1:
                    raise Exception("%s require valid values for alpha" % model)
                for alpha in alphas:
                    model = "PVI(%d)" % int(np.log10(alpha))
                    m_model = BayesianClassifier(model, n_samples, d, c, learning_rate=lr, scale=alpha)
                    bmodels.append(model)
                    models[model] = m_model

            else:
                raise Exception("Unknown model : %s" % model)

        loss_arr = {}
        kl_arr = {}
        mean_arr = {}
        std_arr = {}
        lik_arr = {}

        for model in bmodels:
            loss_arr[model] = []
            kl_arr[model] = []
            mean_arr[model] = []
            std_arr[model] = []
            lik_arr[model] = []

        error_counts = {}
        error_arr = {}

        out_stats = {
            "error": error_arr,
            "loss": loss_arr,
            "kl": kl_arr,
            "mean": mean_arr,
            "std": std_arr,
            "lik": lik_arr
        }

        for model in models:
            error_counts[model] = 0
            error_arr[model] = []

        for i in range(0, n, batch_size):

            X, y = get_minibatch(train_data, train_target, i, batch_size)

            for model in models:

                if i == 0:
                    if model in ["PA", "SGD"]:
                        pred = [0]
                    else:
                        pred = models[model].predict(X)
                else:
                    pred = models[model].predict(X)

                # error logs
                for _p, _y in zip(pred, y):
                    if _p != _y:
                        error_counts[model] += 1
                error_arr[model].append(error_counts[model])

                if model in loss_arr:

                    # if i % stat_interval == 0:
                    #     log_p = models[model].log_predictive_likelihood(test_data, test_target)
                    #     ar = []
                    #     loglik_arr[model].append(log_p[0])

                    lik_arr[model].append(models[model].log_predictive_likelihood(X, y)[0])
                    loss, kl = models[model].fit(X, y, i, iter=n_iter)
                    loss_arr[model].append(loss)
                    kl_arr[model].append(kl)
                    _mean, _std = models[model].get_vars()

                    mean_arr[model].append(str(_mean.tolist()))
                    std_arr[model].append(str(_std.tolist()))

                elif model in ["PA", "SGD"]:
                    models[model].partial_fit(X, y, range(0, c))
                else:
                    models[model].fit(X, y)

            losses_str = "%d" % i
            for model in loss_arr:
                if len(loss_arr[model]) > 0:
                    losses_str += " %s : (%f, %f)" % (model, loss_arr[model][-1], kl_arr[model][-1])
            print(losses_str)

            error_str = "Error Counts ------->"
            for model in models:
                error_str += " %s:%d," % (model, error_counts[model])
            print(error_str)

            print("========================================================")

            if i % stat_rate == 0 or i == n - 1:
                print("writing stats to directory : %s" % directory2)
                for stat in out_stats:
                    path = "%s/%s.csv" % (directory2, stat)
                    stats = out_stats[stat]
                    df = pd.DataFrame(stats)
                    if not os.path.isfile(path):
                        df.to_csv(path)
                    else:
                        df.to_csv(path, mode='a', header=False)

                    for model in stats:
                        stats[model] = []

        error_str = "Final error rate ========>>"
        for model in models:
            error_str += " %s:%f," % (model, error_counts[model] / float(n))
        print(error_str)


if __name__ == '__main__':
    n_iter = 1

    # dataset paths.
    # download data using the google drive links
    airline_dir = '../../../data/drift'
    poker_dir = '../../../data/drift'

    models = ["SSVB", "PVI", "BB-SVB", "SVI"]
    # models = ["PA", "SGD", "AROW"]
    datasets = {
        "airlines": 0.01,
        "poker": 0.01,

    }

    alphas = [1e5]  # for PVI only e.g. [1e5, 1e6, 1e7]
    proportion = None  # to optimize alpha

    run_exp("stats", models, datasets, n_iter, alphas, proportion)
