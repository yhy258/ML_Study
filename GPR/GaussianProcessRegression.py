import math
import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

@jax.jit
def _rbf_kernel_fn_jit(x, xprime):  # x (n x k), xprime (p x k)
    n, k = x.shape
    p, k = xprime.shape
    minus = x.reshape(n, 1, k) - xprime.reshape(1, p, k)  # n, p, k
    upper = jnp.linalg.norm(minus, axis=-1)  # n, p
    return jnp.exp(-(1 / 2) * upper)


def _multivariate_normal(x, mu, cov, eps=1e-5):
    n, k = x.shape

    inv_cov = jnp.linalg.inv(cov)  # k x k
    cov_det = jnp.linalg.det(cov)
    lower = jnp.sqrt((2 * math.pi) ** k * cov_det)

    temp = jnp.dot((x - mu.reshape(1, -1)), inv_cov)  # dot( (n x k), (k x k) )  -> n x k
    upper_temp = jnp.matmul(temp.reshape(n, 1, k), (x - mu.reshape(1, -1)).reshape(n, k,
                                                                                   1))  # matmul ( (n x 1 x k ) , (n x k x 1)) -> (n, 1, 1)
    upper = jnp.exp(-(1 / 2) * upper_temp.squeeze())

    return upper / (lower + eps)

"""
    JIT + gpu의 경우 
    train data : 1000개, test data : 300개일때 기준,
    위의 cpu보다 10배정도 빠르다.
"""
class GaussianProcessRegressionJIT:
    def __init__(self, x, y, kernel_fn):  # train data # y shape : n,
        self.N, self.K = x.shape
        self.x, self.y = x, y
        self.kernel_fn = kernel_fn
        self.train_cov = self.kernel_fn(self.x, self.x)  # n x n

    def inference(self, key, xprime, precision=100):  # p x k
        p, k = xprime.shape
        test_cov = self.kernel_fn(xprime, xprime)  # p x p
        train_test_cov = self.kernel_fn(self.x, xprime)  # n x p

        precision_inv = jax.jit(jnp.linalg.inv)(self.train_cov + jnp.eye(self.N) * (1 / precision))  # n x n
        # p x n
        _pre = jax.jit(jnp.dot)(train_test_cov.T, precision_inv)  # ((p, n) x (n, n)) -> p, n
        new_mu = jax.jit(jnp.dot)(_pre, self.y)  # p x n

        # (p, n) x (n, p) -> (p, p)
        new_cov = test_cov - jax.jit(jnp.dot)(_pre, train_test_cov)

        return jax.random.multivariate_normal(key, new_mu, new_cov), new_mu, new_cov

if __name__ == '__main__':
    def dim1_make_data(func=np.sinc, start=0., end=15., nums=300, sigma_start=0.001, sigma_end=0.1):
        x = np.linspace(start, end, nums)
        y = func(x)
        sigmas = np.linspace(sigma_start, sigma_end, nums)
        t = y + sigmas * np.random.normal(0, 1, nums)
        return x, y, t, sigmas

    x, y, t, sigmas = dim1_make_data(start=-5, end=50, nums=1000)
    uncertainty = 1.96 * sigmas
    plt.figure(figsize=(16, 8))
    plt.fill_between(x, y + uncertainty, y - uncertainty, alpha=0.1)
    plt.scatter(x, t)

    x, y = jnp.array(x), jnp.array(y)
    x = x.reshape(-1, 1)
    test_x = np.linspace(0., 40., 300)
    test_x = jnp.array(test_x).reshape(-1, 1)

    jax.config.update('jax_platform_name', 'gpu')
    gpr = GaussianProcessRegressionJIT(x, y, kernel_fn=_rbf_kernel_fn_jit)

    key = jax.random.PRNGKey(0)
    result, mus, covs = gpr.inference(key, test_x, precision=1000)
    plt.scatter(test_x, result, s=10)
    plt.show()