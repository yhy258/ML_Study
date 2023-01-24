import numpy as np

import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs
import functools

import numpy as np
import jax.numpy as jnp
from  jax.scipy.stats import multivariate_normal
from jax import grad, jit, vmap
from jax import random
from jax import device_put


# import jax; jax.config.update('jax_platform_name', 'cpu')
import jax; jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

key = random.PRNGKey(0) # seed

class GMM:
    def __init__(self, K):
        self.K = K
        self.N = None
        self.D = None

    def _initialize(self, X):
        mean = jnp.mean(X)
        self.pi = jnp.ones(self.K) / self.K  # K
        self.mu = mean + random.normal(key, (self.K, self.D))  # K, D
        self.cov = jnp.stack([jnp.identity(self.D) for _ in range(self.K)])  # K, D, D
        self.gamma = jnp.ones((self.N, self.K)) / self.K  # N, K

    def _tracking(self, string):
        # expectation 문제.
        print(f"{string}", jnp.isnan(self.pi).sum(), jnp.isnan(self.mu).sum(), jnp.isnan(self.cov).sum(),
              jnp.isnan(self.gamma).sum())

    def _get_likelihood(self, X, k):
        return self.pi[k] * multivariate_normal.pdf(X, mean=self.mu[k], cov=self.cov[k])  # n size

    def _expectation(self, X):
        likelihoods = []
        for k in range(self.K):
            likelihoods.append(self._get_likelihood(X, k))
        likelihoods = jnp.stack(likelihoods, axis=-1)  # n x k
        marginalized = jnp.sum(likelihoods, axis=1).reshape(-1, 1)  # n x 1

        self.gamma = likelihoods / marginalized

    def _get_mu(self, X):
        self.mu = jit(jnp.dot)(self.gamma.T, X) / jit(jnp.sum, static_argnums=(1,))(self.gamma, 0).reshape(-1, 1)

    def _get_cov(self, X, under):
        # N, 1, D - 1, K, D -> N, K, D
        m = X.reshape(-1, 1, self.D) - self.mu.reshape(1, -1, self.D)  # N, K, D
        # N, K, 1, 1 x N, K, 1, D -> N, K, 1, D. -> N, K, D, 1 x N, K, 1, D
        gamma_m = jnp.matmul(self.gamma.reshape(self.N, self.K, 1, 1), m.reshape(self.N, self.K, 1, self.D))
        gamma_m_T = jnp.swapaxes(gamma_m, 2, 3)  # N, K, D, 1
        upper = jnp.matmul(gamma_m_T, m.reshape(self.N, self.K, 1, self.D))  # N, K, D, D
        answer = jnp.sum(upper, axis=0) / under.reshape(-1, 1, 1)
        self.cov = answer  # K, D, D

    def _maximization(self, X):
        # mu
        self._get_mu(X)

        # cov
        under = jnp.sum(self.gamma, 0)  # dim : K

        self._get_cov(X, under)

        # pi
        self.pi = jnp.sum(self.gamma, 0) / self.N  # K

    def fit(self, X: np.ndarray, iteration: int):
        self.N, self.D = X.shape

        self._initialize(X)

        for i in range(iteration):
            self._expectation(X)
            # self._tracking("after expectation") 
            self._maximization(X)
            # self._tracking("after maximization") 

    def visualize(self, X):
        fig2 = plt.figure(figsize=(16, 16))
        x, y = np.mgrid[np.min(X[:, 0]) - 0.5:np.max(X[:, 0]) + 0.5:.01,
               np.min(X[:, 1]) - 0.5:np.max(X[:, 1]) + 0.5:.01]
        pos = np.dstack((x, y))

        start = 100 * self.K + 11
        np_mu = np.array(self.mu)
        np_cov = np.array(self.cov)
        for k in range(self.K):
            rv = scipy.stats.multivariate_normal(np_mu[k], np_cov[k])
            ax2 = fig2.add_subplot(start)
            ax2.contourf(x, y, rv.pdf(pos))
            plt.scatter(X[:, 0], X[:, 1])
            start += 1

        plt.show()
