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

seed = 42
np.random.seed(seed)


class M_AROW:

    def __init__(self, nb_class, d, n_iter=1):
        self.w = np.zeros((nb_class, d))
        self.sigma = np.identity(d)
        self.r = 1
        self.nb_class = nb_class
        self.n_iter = n_iter

    def fit(self, X, y):
        for _ in range(self.n_iter):
            w = self.w
            sigma = self.sigma

            # y = ((y - y.min()) * (1/(y.max() - y.min()) * (nb_class-1))).astype('uint8')

            F_t = np.dot(self.w, X.T)

            # compute hinge loss and support vector
            F_s = np.copy(F_t)
            F_s[y] = -np.inf
            s_t = np.argmax(F_s)
            m_t = F_t[y] - F_t[s_t]
            v_t = np.dot(X, np.dot(sigma, X.T))
            l_t = np.maximum(0, 1 - m_t)  # hinge loss

            # update weights
            if l_t > 0:
                beta_t = 1 / (v_t + self.r)
                alpha_t = l_t * beta_t
                self.w[y] = w[y] + (alpha_t * np.dot(sigma, X.T).T)
                self.w[s_t] = w[s_t] - (alpha_t * np.dot(sigma, X.T).T)
                self.sigma = sigma - beta_t * np.dot(np.dot(sigma, X.T), np.dot(X, sigma))

    def predict(self, X):
        return np.argmax(np.dot(self.w, X.T), axis=0)


class AROW:

    def __init__(self, d, n_iter = 1):
        self.w = np.zeros(d)
        self.sigma = np.identity(d)
        self.r = 1
        self.n_iter = n_iter

    def predict(self, X):
        f_t = np.dot(self.w, X.T)
        if f_t >= 0:
            return [1]
        else:
            return [0]

    def fit(self, X, y):
        for i in range(self.n_iter):
            if y > 0:
                y = 1
            else:
                y = -1

            m_t = np.dot(self.w, X.T)
            v_t = np.dot(X, np.dot(self.sigma, X.T))
            l_t = np.maximum(0, 1 - m_t * y)

            if l_t > 0:
                beta_t = 1 / (v_t + self.r)
                alpha_t = l_t * beta_t
                S_x_t = np.dot(X, self.sigma.T)
                self.w = self.w + (alpha_t * y * S_x_t).flatten()
                self.sigma = self.sigma - beta_t * np.dot(S_x_t.T, S_x_t)
