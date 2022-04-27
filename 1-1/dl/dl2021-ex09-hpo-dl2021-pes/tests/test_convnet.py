import os

import torch
import numpy as np
from lib.conv_model import get_conv_model


def test_convnet1():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    convnet1_result = np.array(
        [
            [
                -2.168458,
                -2.2998824,
                -2.4487796,
                -2.3755221,
                -2.3528016,
                -2.2059524,
                -2.3854494,
                -2.1625593,
                -2.3596725,
                -2.3102782,
            ],
            [
                -2.1734958,
                -2.2989097,
                -2.4422445,
                -2.3761733,
                -2.3570125,
                -2.207876,
                -2.383658,
                -2.159416,
                -2.3583608,
                -2.3109708,
            ],
            [
                -2.1789825,
                -2.3000073,
                -2.450742,
                -2.373191,
                -2.348443,
                -2.207066,
                -2.3839264,
                -2.1580656,
                -2.3583944,
                -2.309369,
            ],
            [
                -2.1802406,
                -2.2984173,
                -2.4438827,
                -2.373985,
                -2.3535883,
                -2.2061868,
                -2.380705,
                -2.1625834,
                -2.355207,
                -2.3116062,
            ],
        ]
    )

    torch.manual_seed(0)
    convnet = get_conv_model([2, 2])

    outputs = convnet(torch.rand(4, 1, 28, 28))

    err_msg = "get_conv_model not implemented correctly"
    np.testing.assert_allclose(
        outputs.detach().numpy(), convnet1_result, atol=1e-5, err_msg=err_msg
    )


if __name__ == "__main__":
    test_convnet1()
    print("Test complete.")
