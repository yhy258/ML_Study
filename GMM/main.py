from GMM import GMM
import numpy as np
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    gmm = GMM(3)

    X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=0)

    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)

    gmm.fit(X_aniso, 50)

    print(gmm.mu, gmm.cov, gmm.pi)

    gmm.visualize(X_aniso)