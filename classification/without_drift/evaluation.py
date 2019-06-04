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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from submission.classification.benchmark.arow import M_AROW, AROW
# Dependency imports
from submission.classification.model.bayesian_classification1 import BayesianClassifier

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


def load_20news(num_features, random_state):
    news_train = fetch_20newsgroups(subset="train", data_home=news_dir)
    news_test = fetch_20newsgroups(subset="test", data_home=news_dir)

    train_data = news_train.data
    train_target = news_train.target.astype(np.int32)

    test_data = news_test.data
    test_target = news_test.target.astype(np.int32)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=num_features)
    train_data = vectorizer.fit_transform(train_data)
    test_data = vectorizer.transform(test_data).toarray()

    train_data, train_target = shuffle(train_data, train_target, random_state=random_state)

    return train_data, train_target, test_data, test_target


def load_product_type_dataset(random_state):
    pt_data = pd.read_csv("%s/train.csv" % products_dir)

    data, target = pt_data.ix[:, :-1].values, pt_data.ix[:, -1].astype('category').cat.codes.values
    data, target = shuffle(data, target, random_state=random_state)

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    train_X, test_X, train_y, test_y = train_test_split(data, target.astype(np.int32))
    return train_X, train_y, train_X, train_y


def load_mnist(file_name, random_state):
    mnist = pd.read_csv("%s/%s" % (mnist_dir, file_name))

    data, target = mnist.ix[:, 1:].values, mnist.ix[:, 0].values
    data, target = shuffle(data, target, random_state=random_state)

    # additional column to represent the intercept
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    return data, target.astype(np.int32)


def load_mnist_train_test(random_state):
    train_X, train_y = load_mnist("mnist_train.csv", random_state)
    test_X, test_y = load_mnist("mnist_test.csv", random_state)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    return train_X, train_y, test_X, test_y


def get_minibatch(X, y, current_idx, batch_size=1):
    X, y = X[current_idx:current_idx + batch_size], y[current_idx:current_idx + batch_size]
    return X.toarray() if isinstance(X, csr_matrix) else X, y


def run_exp(outpath, random_states, datasets, model_list, alphas=None, proportion=None):
    id = time.time()
    directory = "%s/%i" % (outpath, int(id))
    stat_rate = 100

    if not os.path.exists(directory):
        os.makedirs(directory)

    for random_state in random_states:
        directory1 = "%s/%i" % (directory, random_state)
        if not os.path.exists(directory1):
            os.makedirs(directory1)

        for dataset in datasets:
            directory2 = "%s/%s" % (directory1, dataset)
            if not os.path.exists(directory2):
                os.makedirs(directory2)

            if dataset == "20news":
                data = load_20news(100000, random_state)
            elif dataset == "mnist":
                data = load_mnist_train_test(random_state)
            elif dataset == "product_type":
                data = load_product_type_dataset(random_state)
            else:
                raise Exception(
                    "Dataset %s is not found. Download data from google drive. Set the correct path to data" % dataset)

            train_data, train_target, test_data, test_target = data
            if proportion is not None:
                end = int(train_data.shape[0] * proportion)
                if end <= 0:
                    raise Exception("Proportion is invalid")
                train_data, train_target = train_data[:end], train_target[:end]

            n, d = train_data.shape
            c = np.unique(train_target).shape[0]

            scale = n / batch_size
            # scale = 10e8

            if dataset == "20news":
                lr = 0.05
            else:
                lr = 0.01

            n_iter = 1

            print("Data set summery : number of samples %i, number of features %i, number of classes %i" % (n, d, c))

            # create models
            models = {}
            bmodels = []
            for model in model_list:
                if model == "AROW":
                    if dataset == "20news":
                        m_model = DummyModel()
                        models[model] = m_model
                    else:
                        if c > 2:
                            # m_model = sol.SOL('arow', c)
                            m_model = M_AROW(c, d)  # TODO verify the arrow implementation
                        else:
                            m_model = AROW(d)
                        models[model] = m_model
                elif model == "SGD":
                    m_model = SGDClassifier(max_iter=1, tol=1)
                    models[model] = m_model

                elif model == "PA":
                    m_model = PassiveAggressiveClassifier(max_iter=1, tol=1)
                    models[model] = m_model

                elif model == "BB-SVB":
                    m_model = BayesianClassifier("BB-SVB", n_samples, d, c, learning_rate=lr)
                    models[model] = m_model
                    bmodels.append("BB-SVB")

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

            loglik_arr = {}
            loss_arr = {}
            kl_arr = {}
            mean_arr = {}
            std_arr = {}
            for model in bmodels:
                loss_arr[model] = []
                kl_arr[model] = []
                mean_arr[model] = []
                std_arr[model] = []
                loglik_arr[model] = []

            error_counts = {}
            error_arr = {}

            out_stats = {
                "error": error_arr,
                "loss": loss_arr,
                "kl": kl_arr,
                "mean": mean_arr,
                "std": std_arr
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
                    if pred[0] != y[0]:
                        error_counts[model] += 1
                    error_arr[model].append(error_counts[model])

                    if model in loss_arr:
                        loss, kl = models[model].fit(X, y, i, iter=n_iter)
                        if i % stat_interval == 0:
                            log_p = models[model].log_predictive_likelihood(test_data, test_target)
                            loglik_arr[model].append(log_p[0])

                        _mean, _std = models[model].get_vars()

                        mean_arr[model].append(str(_mean.tolist()[:10]))
                        std_arr[model].append(str(_std.tolist()[:10]))

                        loss_arr[model].append(loss)
                        kl_arr[model].append(kl)

                    elif model in ["PA", "SGD"]:
                        models[model].partial_fit(X, y, range(0, c))
                    else:
                        models[model].fit(X, y)

                losses_str = "%d" % i
                for model in loss_arr:
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

            # final accuracy computation
            final_acc_arr = {}
            for model in models:
                pred = models[model].predict(test_data)

                if c <= 2:
                    acc = f1_score(test_target, np.array(pred).flatten())
                else:
                    acc = f1_score(test_target, np.array(pred).flatten(), average="micro")

                final_acc_arr[model] = [acc]

                print("Accuracy of %s : %f" % (model, acc))

            final_stats = {
                "accuracy": final_acc_arr,
                "likelihood": loglik_arr,
            }

            for stat in final_stats:
                path = "%s/%s.csv" % (directory2, stat)
                pd.DataFrame(final_stats[stat]).to_csv(path)


if __name__ == '__main__':
    n_samples = 1
    batch_size = 1
    stat_interval = 50

    # dataset paths.
    # download data using the google drive links
    news_dir = '../../../data/classification/'
    mnist_dir = '../../../data/classification/mnist'
    products_dir = '../../../data/classification/product_type'

    models = ["SSVB"]
    datasets = ["product_type"]  # ["product_type", "mnist", "20news"]
    random_states = [42, 19, 7, 13, 25]

    alphas = [1e5]  # for PVI only [1e5, 1e6, 1e7]
    proportion = None  # to optimize alpha
    run_exp("stats", random_states, datasets, models, alphas, proportion)

    # run_exp("all", random_states, datasets, models)

    # 20news - 1e5
    # otto - 1e6
    # mnist - 1e6
