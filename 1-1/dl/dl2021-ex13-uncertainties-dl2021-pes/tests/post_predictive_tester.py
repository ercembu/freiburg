
import numpy as np
from lib.blr import BLR
from tests.results import lbr2d_result_post_predictive_mean, lbr2d_result_post_predictive_std


def post_predictive_test():
    np.random.seed(0)
    Sigma_pre = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu_pre = np.array([0.0, 0.0])
    noise = 1.0
    lbr_2d = BLR(mu_pre=mu_pre, sigma_pre=Sigma_pre, noise=noise)

    lbr_2d._mu_post = [0.25, 0.15]
    lbr_2d._sigma_post = [[0.5, 0.5], [0.5, 0.5]]
    x_mean, x_std = lbr_2d.posterior_predictive(np.random.rand(10, 2))

    np.testing.assert_equal(x_mean.shape[0], 10, err_msg="mean has a wrong shape BLR post_predictive")
    np.testing.assert_equal(x_std.shape[0], 10, err_msg="std. dev. has a wrong shape BLR post_predictive")

    if len(x_mean.shape) == 1:
        x_mean = x_mean.reshape(-1, 1)
    if len(x_std.shape) == 1:
        x_std = x_std.reshape(-1, 1)

    err_msg = "post_predictive mean not implemented correctly"
    np.testing.assert_allclose(
        x_mean,
        lbr2d_result_post_predictive_mean,
        atol=1e-5,
        err_msg=err_msg)

    err_msg = "post_predictive std. dev. not implemented correctly"
    np.testing.assert_allclose(
        x_std,
        lbr2d_result_post_predictive_std,
        atol=1e-5,
        err_msg=err_msg)
