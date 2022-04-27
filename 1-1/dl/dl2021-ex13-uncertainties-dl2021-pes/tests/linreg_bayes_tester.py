
import numpy as np
from lib.blr import BLR
from tests.results import lbr2d_result_mean, lbr2d_result_var
from lib.utilities import data_2d


def linreg_bayes_test():
    np.random.seed(0)
    X_train, y_train = data_2d()
    Sigma_pre = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu_pre = np.array([0.0, 0.0])
    noise = 1.0
    lbr_2d = BLR(mu_pre=mu_pre, sigma_pre=Sigma_pre, noise=noise)
    lbr_2d.linreg_bayes(X_train, y_train)
    mu_post = lbr_2d.Mu_post
    Sigma_post = lbr_2d.Sigma_post

    np.testing.assert_array_equal(
        mu_post.shape,
        [2],
        err_msg="mean has a wrong shape BLR linreg_bayes")
    np.testing.assert_array_equal(
        Sigma_post.shape, [
            2, 2], err_msg="variance has a wrong shape BLR linreg_bayes ")

    err_msg = "linreg_bayes: mean not implemented correctly"

    np.testing.assert_allclose(
        mu_post,
        lbr2d_result_mean,
        atol=1e-5,
        err_msg=err_msg)

    err_msg = "linreg_bayes: sigma not implemented correctly"
    np.testing.assert_allclose(
        Sigma_post.flatten(),
        lbr2d_result_var,
        atol=1e-5,
        err_msg=err_msg)
