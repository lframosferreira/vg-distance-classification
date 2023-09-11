from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import statistics
# Serão 8 classificadores, 1 para cada lead. A moda das classificações vai ser a classificação final


class MyClassifier:
    def __init__(self, n_neighbors: int = 10) -> None:
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
        all_preds = np.zeros(X.shape[1])
        for j, single in enumerate(X):
            pred_for_lead = np.zeros(8)
            for i, clf in enumerate(self.clfs):
                pred_for_lead[i] = clf.predict(single)[0]
            all_preds[j] = statistics.mode(pred_for_lead)
        return all_preds
