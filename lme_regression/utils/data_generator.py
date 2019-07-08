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
import pandas as pd


class DataGenerator:

    def __init__(self, num_fixed_dim, sigma_e=0.5, sigma_b=1.0):
        self.num_dim = num_fixed_dim
        self.sigma_e = sigma_e
        self.sigma_b = sigma_b
        self.w = np.random.normal(0, 5, self.num_dim)
        # print(self.w)

        # print(self.w)

    def generate_data(self, n_samples_per_cluster):
        # ~~~~~~~~~ Fixed Effect Generation ~~~~~~~~~~ #
        # draw the fixed effects features for each samples (3 dims independent)
        X = np.random.normal(loc=0, scale=1.0, size=(sum(n_samples_per_cluster), self.num_dim))
        X_df = pd.DataFrame(X)

        # generate the fixed effect response
        g = np.dot(X, self.w)

        # ~~~~~~~~~ Random Effect Generation ~~~~~~~~~~ #
        # Create the number of clusters from vector of samples per cluster
        n_clusters = len(n_samples_per_cluster)

        # Create vector of cluster_id for each sample
        Z = []
        for i in range(0, n_clusters):
            cluster_id = i
            n_samples = n_samples_per_cluster[i]
            zi = cluster_id * np.ones(n_samples, dtype=np.int8)  # want cluster id to be int
            Z.extend(zi)

        # one hot encode it for easier addition to get response
        # Z_ohe = pd.get_dummies(Z)

        # create groups partition and random intercept value
        clusters_df = pd.Series(Z)
        Z_df = pd.DataFrame(np.ones(len(Z)))

        if not hasattr(self, 'b'):
            # draw the random effect bias for each cluster
            self.b = np.random.normal(loc=5, scale=5.0, size=n_clusters)
            self.scale_b = np.abs(np.random.normal(loc=0, scale=self.sigma_b, size=n_clusters))

        # new random effect
        re = np.array([np.random.normal(loc=self.b[z], scale=self.scale_b[z]) for z in Z])

        # generate the random effect response
        # re = Z_ohe.dot(self.b)

        # ~~~~~~~~~ Noise Generation ~~~~~~~~~~ #
        eps = np.random.normal(loc=0, scale=self.sigma_e, size=sum(n_samples_per_cluster))

        # ~~~~~~~~~ Response Generation ~~~~~~~~~~ #
        # add fixed effe
        # ct, random effect, and noise to get final response
        y = g + re + eps
        y_df = pd.Series(y)

        # merge all the separate matrices into one matrix
        merged_df = pd.concat((y_df, X_df, Z_df, clusters_df), axis=1)
        merged_df.columns = ['y'] + ['X_%i' % i for i in range(self.num_dim)] + ['Z', 'cluster']
        return merged_df

    def generate_train_test_splits(self, num_clusters_train, num_clusters_test):

        train = self.generate_data(num_clusters_train)
        test = self.generate_data(num_clusters_test)

        return train, test

    def add_drift(self, drift=0.1, drift_b=False):
        if drift_b:
            if not hasattr(self, 'b'):
                raise Exception()
            self.b = self.b + np.random.normal(0, drift, self.b.shape)
            self.scale_b = np.abs(self.scale_b + np.random.normal(0, drift, self.scale_b.shape))

        self.w = self.w + np.random.normal(0, drift, self.w.shape)
