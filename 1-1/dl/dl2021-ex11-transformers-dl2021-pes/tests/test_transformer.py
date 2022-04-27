"""Script to test implementation of TransformerModel"""

import numpy as np
import torch
from lib.transformer import TransformerModel


def test_transformer_no_enc_dec():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    test_model = TransformerModel(4, 4, 2, 4, 0, 0, 2)

    torch.set_printoptions(precision=6)

    decoding, _ = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [0.079619, 0.887964, 0.172323, -0.240035],
                [-0.286800, 0.139949, 0.133315, -0.234130],
                [0.062089, 0.870800, 0.255983, -0.314032],
            ],
            [
                [0.046210, 0.850558, 0.280015, -0.257313],
                [-0.079682, 0.475454, 0.405175, -0.532975],
                [-0.108757, 0.178055, 0.799308, -0.401608],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )


def test_transformer_enc():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    test_model = TransformerModel(4, 4, 2, 4, 1, 0, 2)

    torch.set_printoptions(precision=6)

    decoding, _ = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [7.263068e-01, -1.014227e-01, -3.823230e-02, -1.762019e-01],
                [5.700462e-01, -3.183520e-01, -6.464455e-01, -3.496110e-04],
                [4.366536e-01, 6.375909e-03, -2.316464e-01, -2.000339e-01],
            ],
            [
                [4.548398e-01, 7.573038e-02, -3.700223e-01, -1.584404e-01],
                [-2.621710e-04, -2.487601e-01, -3.381877e-01, -3.074347e-01],
                [8.121517e-02, -4.132197e-01, -4.981286e-01, -3.680885e-01],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )


def test_transformer_dec():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    test_model = TransformerModel(4, 4, 2, 4, 0, 1, 2)

    torch.set_printoptions(precision=6)

    decoding, attention = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [-0.437948, 0.329755, 0.973106, 0.261721],
                [-0.530101, 0.834532, 0.486800, 0.717390],
                [-0.475184, 0.174263, 0.988986, 0.120098],
            ],
            [
                [-0.352493, 0.164801, 1.035215, 0.124756],
                [-0.830256, 0.597120, 0.487859, 0.471934],
                [-0.533838, 0.798108, 0.560142, 0.681227],
            ],
        ]
    )

    expected_attention = torch.tensor(
        [
            [
                [[0.499935, 0.500065], [0.499853, 0.500147], [0.499967, 0.500033]],
                [[0.500039, 0.499961], [0.499985, 0.500015], [0.499979, 0.500021]],
            ],
            [
                [[0.499996, 0.500004], [0.500026, 0.499974], [0.499996, 0.500004]],
                [[0.499981, 0.500019], [0.500057, 0.499943], [0.500020, 0.499980]],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )
    np.testing.assert_almost_equal(
        attention.detach().numpy(),
        expected_attention,
        decimal=5,
        err_msg=err_msg,
    )


def test_transformer():
    torch.manual_seed(0)

    x1 = torch.randn(2, 2, 4)
    x2 = torch.randn(2, 3, 4)

    test_model = TransformerModel(4, 4, 2, 4, 2, 2, 2)

    torch.set_printoptions(precision=6)

    decoding, attention = test_model(x1, x2)

    expected_decoding = torch.tensor(
        [
            [
                [0.125010, -0.053467, 0.394388, -0.581187],
                [0.106670, -0.050776, 0.465780, -0.603126],
                [0.237060, -0.090937, 0.254279, -0.541851],
            ],
            [
                [0.287407, -0.110055, 0.099694, -0.477356],
                [0.010707, -0.072476, 0.986136, -0.714397],
                [-0.018049, -0.020663, 0.566531, -0.598349],
            ],
        ]
    )

    expected_attention = torch.tensor(
        [
            [
                [[0.500002, 0.499998], [0.499991, 0.500009], [0.500042, 0.499958]],
                [[0.500040, 0.499960], [0.499928, 0.500072], [0.499959, 0.500041]],
            ],
            [
                [[0.499720, 0.500280], [0.499749, 0.500251], [0.499667, 0.500333]],
                [[0.499718, 0.500282], [0.500037, 0.499963], [0.499858, 0.500142]],
            ],
        ]
    )

    err_msg = "TransformerModel forward pass not implemented correctly"

    np.testing.assert_almost_equal(
        decoding.detach().numpy(), expected_decoding, decimal=5, err_msg=err_msg
    )
    np.testing.assert_almost_equal(
        attention.detach().numpy(),
        expected_attention,
        decimal=5,
        err_msg=err_msg,
    )


if __name__ == "__main__":
    test_transformer_no_enc_dec()
    test_transformer_enc()
    test_transformer_dec()
    test_transformer()
    print("Test complete.")
