
import numpy as np


class SMOTE:
    def __init__(self, k_neighbors=5, random_state=None):
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")

        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D label array.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        classes, counts = np.unique(y, return_counts=True)

        if len(classes) != 2:
            raise ValueError("This version supports binary classification only.")

        minority_class = classes[np.argmin(counts)]
        minority_indices = np.flatnonzero(y == minority_class)
        X_minority = X[minority_indices]

        n_to_generate = int(counts.max() - counts.min())

        if n_to_generate == 0:
            return X.copy(), y.copy()

        if len(X_minority) <= self.k_neighbors:
            raise ValueError(
                "k_neighbors must be smaller than the minority-class count."
            )

        rng = np.random.default_rng(self.random_state)
        synthetic_samples = []

        for _ in range(n_to_generate):
            # Randomly choose a minority sample as the starting point.
            base_index = rng.integers(len(X_minority))
            x_i = X_minority[base_index]

            # Calculate distances to all minority samples.
            distances = np.sqrt(
                np.sum((X_minority - x_i) ** 2, axis=1)
            )

            # Exclude the starting sample itself.
            distances[base_index] = np.inf

            # Select its k nearest neighbours.
            neighbor_indices = np.argsort(distances)[:self.k_neighbors]

            # Randomly select one of those neighbours.
            neighbor_index = rng.choice(neighbor_indices)
            x_nn = X_minority[neighbor_index]

            # Generate a synthetic point through interpolation.
            gap = rng.random()
            x_new = x_i + gap * (x_nn - x_i)

            synthetic_samples.append(x_new)

        X_synthetic = np.asarray(synthetic_samples)
        y_synthetic = np.full(n_to_generate, minority_class)

        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])

        return X_resampled, y_resampled
