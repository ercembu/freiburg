import numpy as np
import torch

from lib.counting import CountingModel


def test_counting_model_forward():
    torch.manual_seed(0)

    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 2)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)

    expected_out = np.array(
        [
            [[0.401484, 0.758529], [0.117838, 0.422515]],
            [[0.401519, 0.758571], [0.392864, 0.748319]],
        ]
    )
    expected_weights = np.array([[[0.574604], [0.425396]], [[0.549879], [0.450121]]])

    err_msg = "Attention forward pass not implemented correctly"
    np.testing.assert_allclose(
        out.detach().numpy(), expected_out, err_msg=err_msg, rtol=1e-5
    )
    np.testing.assert_allclose(
        attention_weights.detach().numpy(), expected_weights, err_msg=err_msg, rtol=1e-5
    )


def test_counting_model_backward():
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)
    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 2)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)
    loss_function = torch.nn.MSELoss()
    expected_out = torch.tensor(
        [
            [[0.40, 0.75], [0.11, 0.42]],
            [[0.40, 0.75], [0.39, 0.74]],
        ]
    )
    torch.set_printoptions(precision=9)
    loss = loss_function(out, expected_out)
    model.zero_grad()
    loss.backward()

    expected_grads = np.array(
        [
            [
                [
                    [-6.234770808e-06, 4.228021862e-05],
                    [-4.239410628e-03, 4.203408025e-03],
                ]
            ],
            [0.002824201, -0.001926283],
            [-0.002390039, -0.000696722],
            [[1.079780304e-05, 1.931806219e-05], [1.684605013e-05, 3.013882269e-05]],
            [2.651639079e-05, 4.136920325e-05],
            [[-0.000775022, -0.001222415], [0.000775030, 0.001222428]],
            [-0.001878337, 0.001878356],
            [[0.000458526, -0.000458524], [-0.005624017, 0.005624016]],
            [0.003426258, 0.006983712],
        ]
    )

    err_msg = "Attention forward pass not implemented correctly"

    for i, param in enumerate(model.parameters()):
        np.testing.assert_allclose(
            param.grad.detach().numpy(), expected_grads[i], rtol=1e-3, err_msg=err_msg
        )


if __name__ == "__main__":
    test_counting_model_forward()
    test_counting_model_backward()
    print("Test complete.")
