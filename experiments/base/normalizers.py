import numpy as np


class Normalizer:
    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class EquivariantNormalizer:
    def fit(self, X):
        self.std = np.nanstd(X)

    def transform(self, X):
        return X / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def get_normalizer(model_name):
    if model_name == "lgatr":
        return EquivariantNormalizer()
    return Normalizer()
