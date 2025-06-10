from syssim.core.fault import Fault
import numpy as np

class DiagonalInertiaPerturbFault(Fault):
    """Fault that perturbs each axis of a 3x3 diagonal inertia matrix by a random amount."""
    def __init__(self, name: str, stddev: float = 0.1, seed: int = None):
        """
        Args:
            name (str): Name of the fault.
            stddev (float): Standard deviation of the random perturbation (fractional, e.g., 0.01 for 1%).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(name)
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)
        # Generate random perturbation factors for each axis
        self._perturbation = 1.0 + self.rng.normal(0, self.stddev, size=3)

    def action(self, value: np.ndarray):
        """Perturb the diagonal elements of a 3x3 diagonal inertia matrix by a random amount."""
        assert isinstance(value, np.ndarray), "Value must be a numpy ndarray."
        assert value.shape == (3, 3), "Inertia matrix must be 3x3."
        assert np.allclose(value, np.diag(np.diagonal(value))), "Matrix must be diagonal."
        perturbed_diag = np.diagonal(value) * self._perturbation
        return np.diag(perturbed_diag)
