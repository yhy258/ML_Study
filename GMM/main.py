# Jax version

from GMM import GMM
import timeit
import jax.numpy as jnp
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    gmm = GMM(3)

    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

    transformation = jnp.array([[0.60834549, -0.63667341], [-0.40887718, 0.85253229]], jnp.float64)
    X_aniso = jnp.dot(X, transformation)
    print(X_aniso.dtype)
    %timeit gmm.fit(X_aniso, 20)

    print(gmm.mu, gmm.cov, gmm.pi)

    gmm.visualize(X_aniso)