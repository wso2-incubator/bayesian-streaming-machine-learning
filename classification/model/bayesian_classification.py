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

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

tfd = tfp.distributions


class BayesianClassifier:

    def __init__(self, name, n_samples, n_dim, n_class, learning_rate=0.005, scale=None, normalization=1.0):
        """
        initiating the model
        :param name : type of the model [SSVB, SSVB-CF, SSVB-CV, SVB]
        :param n_samples: number of samples draw during MC integration
        :param n_dim: number of dimensions
        :param n_class: number of classes
        :param learning_rate: optimizer initial learning rate (does not use any decaying learning rate explicitly)
        :param scale: Keep 'None' for SSVB. otherwise use an integer which define the scale of the log likelihood term
        """

        all_names = ["BB-SVB", "SSVB", "SSVB-CV", "SVI"]
        self.name = name.upper()

        if self.name not in all_names:
            if self.name.startswith("PVI"):
                self.name = "SVI"
            else:
                raise Exception("Invalid model name. Should be one of " + all_names)
        if self.name == "SVI" and scale is None:
            raise Exception("For SVB the scale cannot be none")

        if self.name != "SSVB":
            if normalization < 0:
                print ("Normalization should be greater than 0")
                normalization = 1.0
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(seed)

            with tf.variable_scope(self.name):
                self.inputs = tf.placeholder(tf.float32, [None, n_dim], "input_X")
                self.labels = tf.placeholder(tf.int32, [None], "input_y")

                step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.float32)
                self.step_increment = tf.assign(step, step + 1)
                self.step_reset = tf.assign(step, 0)

                # building the inference network
                likelihood, qw = self.softmax_regression(n_class)

                # defining the priors
                self.loc_pw = tf.Variable(tf.zeros([n_dim, n_class]), trainable=False, name="loc_pw")
                self.scale_pw = tf.Variable(tf.ones([n_dim, n_class]), trainable=False, name="scale_pw")
                pw = tfd.Normal(loc=self.loc_pw, scale=self.scale_pw)

                self.pw, self.qw = pw, qw
                # loss computation
                self.p_log_lik_arr = [0.0] * n_samples  # likelihood
                for s in range(n_samples):  # mc-integration of the expectation of log likelihood
                    self.p_log_lik_arr[s] = tf.reduce_sum(likelihood.log_prob(self.labels))

                self.p_log_lik = tf.reduce_mean(self.p_log_lik_arr)

                if self.name != "SVI":
                    scaler = 1.0
                    if self.name == "SSVB":
                        scaler = tf.multiply(step, normalization)
                    print("Using %s" % self.name)
                    self.kl_penalty = tf.reduce_sum([tf.reduce_sum(tfd.kl_divergence(qw, pw))]) / scaler
                    elbo_loss = -(self.p_log_lik - self.kl_penalty)
                else:
                    print("Using SVB with scale : %4f" % scale)
                    self.kl_penalty = tf.reduce_sum([tf.reduce_sum(tfd.kl_divergence(qw, pw))]) / scale
                    elbo_loss = -(self.p_log_lik - self.kl_penalty)

                self.elbo_loss = elbo_loss

                opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            with tf.variable_scope(self.name, reuse=True):

                var_list = [tf.get_variable("loc_qw"), tf.get_variable("scale_qw")]

            with tf.variable_scope(self.name):
                self.train_op = opt.minimize(self.elbo_loss, var_list=var_list)

                self.grads = opt.compute_gradients(-self.p_log_lik, var_list=var_list)

            with tf.variable_scope(self.name, reuse=True):

                # TODO try using forward pass (sampling)
                # Predictions
                self.probs = tf.nn.softmax(tf.tensordot(self.inputs, qw.loc, axes=1))
                self.p_density = [tf.nn.softmax(tf.tensordot(self.inputs, qw.loc, axes=1)) for _ in range(1000)]

                self.pred_softmax = tf.argmax(self.probs, axis=1)

                # log predictive probability
                self.log_pred_p = tf.reduce_mean([likelihood.log_prob(self.labels) for _ in range(10000)])

                if self.name != "SVI":
                    if self.name == "BB-SVB":
                        self.update_qw = self.loc_pw.assign(qw.loc), self.scale_pw.assign(qw.scale)
                    else:
                        self.update_qw = self.loc_pw.assign(qw.loc), \
                                         self.scale_pw.assign(tf.ones([n_dim, n_class]))

                init = tf.global_variables_initializer()

        tf.reset_default_graph()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

        # # initializing the variables
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

    def fit(self, X, y, n, iter=1):
        """
        training the model
        :param X:
        :param y:
        :return:
        """
        if iter < 1:
            iter = 1

        for i in range(iter):
            loss_value, kl = self.sess.run([self.elbo_loss, self.kl_penalty],
                                           feed_dict={
                                               self.inputs: X,
                                               self.labels: y,
                                           })

            self.sess.run([self.train_op], feed_dict={
                self.inputs: X,
                self.labels: y,
            })

        if self.name != "SVI":
            self.sess.run(self.update_qw)
            self.sess.run(self.step_increment)

        return loss_value, kl

    def predict(self, X):
        """
        predict using the model
        :param X:
        :return:
        """

        pred_softmax = self.sess.run([self.pred_softmax],
                                     feed_dict={
                                         self.inputs: X
                                     })

        return pred_softmax

    def pred_proba(self, X):
        """
        predict using the model
        :param X:
        :return:
        """

        probs = self.sess.run([self.probs], feed_dict={
            self.inputs: X})

        return probs

    def predict_with_uncertainty(self, X):
        """
        predict using the model
        :param X:
        :return: prediction and uncertainty
        """

        pred_softmax = self.sess.run(self.p_density,
                                     feed_dict={
                                         self.inputs: X
                                     })

        c = np.argmax(np.mean(pred_softmax, axis=0).flatten())
        unc = np.std(pred_softmax, axis=0).flatten()[c]
        return [c], unc

    def log_predictive_likelihood(self, X, y):
        """
        evaluate the convergence using log predictive likelihood of the model
        :param X:
        :param y:
        :return:
        """

        log_predictive_likelihood = self.sess.run([self.log_pred_p],
                                                  feed_dict={
                                                      self.inputs: X,
                                                      self.labels: y
                                                  })

        return log_predictive_likelihood

    def softmax_regression(self, c):
        """
        Logistic regression inference network
        :param inputs:
        :return:
        """

        d = self.inputs.shape[-1]

        qw = tfd.Normal(loc=tf.get_variable("loc_qw", [d, c], ),
                        scale=tf.nn.softplus(
                            tf.get_variable("scale_qw", [d, c], )))

        y = tfd.Categorical(logits=tf.tensordot(self.inputs, qw.sample(seed=seed), axes=1))

        return y, qw

    def fit_only(self, X, y):
        """
        training the model
        :param X:
        :param y:
        :return:
        """

        self.sess.run([self.train_op], feed_dict={
            self.inputs: X,
            self.labels: y
        })

    def get_vars(self):
        """
        :return:
        """
        return self.sess.run([self.qw.loc, self.qw.scale])

    def get_grads(self, X, y):
        return self.sess.run(self.grads, feed_dict={self.inputs: X, self.labels: y})[0][0]
