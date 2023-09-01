from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy

# Serão 8 classificadores, 1 para cada lead. A moda das classificações vai ser a classificação final


class MyClassifier:
    def __init__(self, n_neighbors: int) -> None:
        self.clfs = [
            KNeighborsClassifier(
                n_neighbors=n_neighbors, n_jobs=-1, metric="precomputed"
            )
            for _ in range(8)
        ]

    # X e y são 3d e 1d, respectivamente. Contêm as distâncias entre cada lead, mas as classes são a mesma pra todas
    def fit(self, X, y):
        for i in range(8):
            self.clfs[i].fit(X[i], y)

    def predict(self, X):
        pred = np.zeros(8)
        for i, clf in enumerate(self.clfs):
            pred[i] = clf.predict(X)
        return scipy.stats.mode(pred)
