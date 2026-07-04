"""gpflow implementation of the GPBackend interface.

All gpflow / TensorFlow calls used by the algorithm live here (moved verbatim from
GP_Emulator), so importing the rest of the package does not require gpflow/TF. The
numerics are identical to the pre-extraction code.
"""
import numpy as np
import gpflow
import tensorflow as tf
from tensorflow_probability import bijectors as tfb

from .base import GPBackend


class GpflowBackend(GPBackend):
    name = "gpflow"

    def configure(self):
        tf.compat.v1.get_default_graph()
        gpflow.config.set_default_float(np.float64)

    def make_bounded_parameter(self, low, high, initial_value):
        sigmoid = tfb.Sigmoid(
            low=tf.cast(low, dtype=tf.float64), high=tf.cast(high, dtype=tf.float64)
        )
        return gpflow.Parameter(initial_value, transform=sigmoid, dtype=tf.float64)

    def build_model(self, data, kernel_value, lenscls, tau, white_var,
                    fix_lengthscale, fix_outputscale, noise_variance=1e-5):
        if kernel_value == 3:
            gpKernel = gpflow.kernels.SquaredExponential(
                variance=tau, lengthscales=lenscls
            )
        elif kernel_value == 2:
            gpKernel = gpflow.kernels.Matern32(variance=tau, lengthscales=lenscls)
        else:
            gpKernel = gpflow.kernels.Matern52(variance=tau, lengthscales=lenscls)
        # Add White kernel
        gpKernel = gpKernel + gpflow.kernels.White(variance=white_var)

        # Build GP model
        gp_model = gpflow.models.GPR(data, kernel=gpKernel, noise_variance=noise_variance)
        # Select whether the likelihood variance is trained
        gpflow.utilities.set_trainable(gp_model.likelihood.variance, False)
        if fix_lengthscale:
            gpflow.utilities.set_trainable(gp_model.kernel.kernels[0].lengthscales, False)
        if fix_outputscale:
            gpflow.utilities.set_trainable(gp_model.kernel.kernels[0].variance, False)
        return gp_model

    def train(self, model):
        # Build optimizer
        optimizer = gpflow.optimizers.Scipy()
        # Fit GP to training data.
        # compile=False (eager) is REQUIRED for reliability + determinism: the default
        # compile=True traces a tf.function whose gradient intermittently raises
        # "InvalidArgumentError: Expected tensor of type double but got type float" under
        # CPU contention (training data + params are already float64, so this is a TF
        # graph/threading issue, not a data-dtype bug). compile=False never crashes and is
        # bit-for-bit reproducible. The golden regression baseline is captured with this
        # setting. See refactor_notes.md (Phase 0 lesson 1 / Phase 2a).
        aux = optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options={"maxiter": 10**9},
            method="L-BFGS-B",
            compile=False,
        )
        training_loss = model.training_loss().numpy()
        return bool(aux.success), training_loss

    def get_hyperparameters(self, model):
        outputscl_final = float(model.kernel.kernels[0].variance.numpy())
        lenscl_final = model.kernel.kernels[0].lengthscales.numpy()
        noise_final = float(model.kernel.kernels[1].variance.numpy())
        return [lenscl_final, noise_final, outputscl_final]

    def make_posterior(self, model):
        return model.posterior()

    def predict_f(self, posterior, eval_points, full_cov=True):
        eval_points_tf = tf.convert_to_tensor(eval_points)
        gp_mean_scl, gp_covar_scl = posterior.predict_f(eval_points_tf, full_cov=full_cov)
        gp_mean_scl = gp_mean_scl.numpy()
        gp_covar_scl = np.squeeze(gp_covar_scl.numpy(), axis=0)
        return gp_mean_scl, gp_covar_scl
