import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture import _gaussian_mixture as gm


def _estimate_gaussian_parameters(X, means, resp, reg_covar, covariance_type):
    """is based on the original function"""
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    covariances = {
        "full": gm._estimate_gaussian_covariances_full,
        "tied": gm._estimate_gaussian_covariances_tied,
        "diag": gm._estimate_gaussian_covariances_diag,
        "spherical": gm._estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, covariances


class FixedMeansGM(GaussianMixture):
    def _check_parameters(self, X):
        super()._check_parameters(X)
        assert self.means_init is not None, "means_init should be provided"

    def _m_step(self, X, log_resp):
        """is taken from the original _m_step"""
        self.weights_, self.covariances_ = _estimate_gaussian_parameters(
            X, self.means_, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = gm._compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
