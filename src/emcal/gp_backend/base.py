"""Abstract GP-backend interface.

The GPBO algorithm (``GP_Emulator`` in ``GPBO_Classes_New``) owns all orchestration —
data scaling, the multi-restart training loop, hyperparameter initialization — and
delegates only the library-specific Gaussian-process primitives to a ``GPBackend``.
Model / parameter / posterior objects returned by a backend are opaque to the caller
and are only ever passed back to the same backend instance.

This seam lets the GP library (currently gpflow) be swapped without touching the
algorithm. Only gpflow is implemented today; a second backend would implement this
same interface.
"""
from abc import ABC, abstractmethod


class GPBackend(ABC):
    name = "abstract"

    @abstractmethod
    def configure(self):
        """One-time global configuration (e.g. set default float precision)."""

    @abstractmethod
    def make_bounded_parameter(self, low, high, initial_value):
        """Return a trainable parameter constrained to (low, high) via a sigmoid transform."""

    @abstractmethod
    def build_model(self, data, kernel_value, lenscls, tau, white_var,
                    fix_lengthscale, fix_outputscale, noise_variance=1e-5):
        """Build an (untrained) GP regression model.

        data: (X, y) numpy tuple. kernel_value: 3=RBF, 2=Matern32, else Matern52.
        lenscls/tau/white_var: parameters from make_bounded_parameter.
        fix_lengthscale/fix_outputscale: if True, mark that hyperparameter non-trainable.
        """

    @abstractmethod
    def train(self, model):
        """Optimize the model hyperparameters. Returns (success: bool, training_loss: float)."""

    @abstractmethod
    def get_hyperparameters(self, model):
        """Return trained [lengthscale (np.ndarray), noise (float), outputscale (float)]."""

    @abstractmethod
    def make_posterior(self, model):
        """Return a posterior object for fast prediction."""

    @abstractmethod
    def predict_f(self, posterior, eval_points, full_cov=True):
        """Predict latent mean and (co)variance at eval_points.

        Returns (mean: np.ndarray, covar: np.ndarray) with the leading singleton
        dimension of the covariance already squeezed out.
        """
