import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
"""
    Train dataset : N x D
    mu : D
    cov : D
    cluster num : K
"""


class GMM:
    def __init__(self, K):
        self.K = K
        self.D = None

    def _initialize(self, X):
        mean = np.mean(X)
        self.pi = np.ones(self.K) / self.K  # K
        self.mu = mean + np.random.randn(self.K, self.D)  # K, D
        self.cov = np.stack([np.identity(self.D) for _ in range(self.K)])  # K, D, D
        self.gamma = np.ones((self.N, self.K)) / self.K  # N, K

    def _get_joint(self, X, n, k):
        return self.pi[k] * multivariate_normal.pdf(X[n], mean=self.mu[k], cov=self.cov[k])

    def _marginalize(self, X, n):
        ans = 0
        for k in range(self.K):
            ans += self._get_joint(X, n, k)
        return ans

    def _expectation(self, X):
        for n in range(self.N):
            evidence = self._marginalize(X, n)
            for k in range(self.K):
                self.gamma[n, k] = self._get_joint(X, n, k) / evidence

    def _maximization(self, X):
        # mu
        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, 0).reshape(-1, 1)  # K, D

        # cov
        under = np.sum(self.gamma, 0)
        for k in range(self.K):
            m = X - self.mu[0].reshape(1, -1)
            mid = np.matmul(m.reshape(self.N, self.D, 1), m.reshape(self.N, 1, self.D))
            self.cov[k] = np.einsum("i,ijk->jk", *[self.gamma[:, k], mid]) / under[k]

        # pi
        self.pi = np.sum(self.gamma, 0) / self.N  # K

    def fit(self, X: np.ndarray, iteration: int):
        self.N, self.D = X.shape
        print(self.N, self.D)

        self._initialize(X)

        for i in range(iteration):
            self._expectation(X)
            self._maximization(X)

    def visualize(self, X):
        fig2 = plt.figure(figsize=(16, 16))
        x, y = np.mgrid[np.min(X[:, 0]) - 0.5:np.max(X[:, 0]) + 0.5:.01,
               np.min(X[:, 1]) - 0.5:np.max(X[:, 1]) + 0.5:.01]
        pos = np.dstack((x, y))

        start = 100 * self.K + 11
        for k in range(self.K):
            rv = multivariate_normal(self.mu[k], self.cov[k])
            ax2 = fig2.add_subplot(start)
            ax2.contourf(x, y, rv.pdf(pos))
            plt.scatter(X[:, 0], X[:, 1])
            start += 1

        plt.show()


