import numpy as np
import torch

from lib.attention import attention_function


def test_attention_function():
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)
    dk = 2
    q = torch.rand(1, 3, dk)
    k = torch.rand(1, 3, dk)
    v = torch.rand(1, 3, dk)
    attention, weights = attention_function(q, k, v, dk)

    expected_attention = np.array(
        [[[0.385528, 0.572083], [0.278429, 0.400367], [0.349924, 0.514942]]]
    )

    expected_weights = np.array(
        [
            [
                [0.409771, 0.392774, 0.374029],
                [0.237701, 0.259151, 0.282314],
                [0.352527, 0.348075, 0.343657],
            ]
        ]
    )

    err_msg = "attention_function not implemented correctly"
    np.testing.assert_allclose(
        attention.detach().numpy(), expected_attention, err_msg=err_msg, rtol=1e-5
    )
    np.testing.assert_allclose(
        weights.detach().numpy(), expected_weights, err_msg=err_msg, rtol=1e-5
    )


if __name__ == "__main__":
    test_attention_function()
    print("Test complete.")
