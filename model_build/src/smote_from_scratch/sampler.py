
import numpy as np


class SMOTE:
    """
    A from-scratch implementation of SMOTE for binary classification.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest minority-class neighbours to consider.

    random_state : int or None, default=None
        Seed for reproducible synthetic sample generation.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        if (
            isinstance(k_neighbors, bool)
            or not isinstance(k_neighbors, (int, np.integer))
            or k_neighbors < 1
        ):
            raise ValueError("k_neighbors must be a positive integer.")

        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _find_k_neighbors(self, X_minority, sample_index):
        """
        Return the indices of the k nearest neighbours of a minority sample,
        excluding the sample itself.
        """
        distances = np.sqrt(
            np.sum(
                (X_minority - X_minority[sample_index]) ** 2,
                axis=1
            )
        )

        # Exclude the reference sample from its own neighbour list.
        distances[sample_index] = np.inf

        # Return indices sorted by ascending distance.
        neighbor_indices = np.argsort(distances)[:self.k_neighbors]

        return neighbor_indices

    def fit_resample(self, X, y):
        """
        Generate synthetic minority-class samples and return the resampled
        feature matrix and label array.

        This initial implementation supports binary classification and
        automatically balances the minority class with the majority class.
        """
        # Convert input data into NumPy arrays.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Validate input dimensions.
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D label array.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one sample and one feature."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X must contain only finite numeric values."
            )

        # Count observations belonging to each class.
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) != 2:
            raise ValueError(
                "This version supports binary classification only."
            )

        # Identify the minority class.
        minority_class = classes[np.argmin(counts)]

        # Extract only minority-class observations for neighbour searches.
        minority_indices = np.flatnonzero(y == minority_class)
        X_minority = X[minority_indices]

        # Generate enough samples to match the majority-class count.
        n_to_generate = int(counts.max() - counts.min())

        # If already balanced, return copies of the original arrays.
        if n_to_generate == 0:
            return X.copy(), y.copy()

        # At least k_neighbors + 1 minority samples are needed.
        if len(X_minority) <= self.k_neighbors:
            raise ValueError(
                "k_neighbors must be smaller than the minority-class count."
            )

        rng = np.random.default_rng(self.random_state)
        synthetic_samples = []

        # Generate the required number of synthetic observations.
        for _ in range(n_to_generate):

            # Randomly choose a minority sample as the starting point.
            base_index = rng.integers(len(X_minority))
            x_i = X_minority[base_index]

            # Find its k nearest minority-class neighbours.
            neighbor_indices = self._find_k_neighbors(
                X_minority,
                base_index
            )

            # Randomly choose one of those neighbours.
            neighbor_index = rng.choice(neighbor_indices)
            x_nn = X_minority[neighbor_index]

            # Interpolate between the two minority samples.
            gap = rng.random()
            x_new = x_i + gap * (x_nn - x_i)

            synthetic_samples.append(x_new)

        # Convert synthetic samples into arrays.
        X_synthetic = np.asarray(synthetic_samples, dtype=float)
        y_synthetic = np.full(n_to_generate, minority_class)

        # Append synthetic samples while preserving original observations.
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])

        return X_resampled, y_resampled
