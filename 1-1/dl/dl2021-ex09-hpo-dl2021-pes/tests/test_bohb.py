import os
import numpy as np
import torch
from lib.worker import PyTorchWorker

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB


def test_pytorch_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker_config_space_lr = 0.0005547119471592127
    worker_end_res = [0.1005, 0.10693359375, 0.099609375]
    torch.manual_seed(0)
    worker = PyTorchWorker(run_id="0")
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    config_lr = config["lr"]
    err_msg = "Worker config_space not implemented correctly"
    np.testing.assert_allclose(
        config_lr, worker_config_space_lr, atol=1e-5, err_msg=err_msg
    )

    min_budget = 1
    max_budget = 1
    working_dir = os.curdir
    host = "localhost"
    port = 0
    run_id = "bohb_run_1"
    n_bohb_iterations = 1
    try:
        # Start a nameserver #####
        ns = hpns.NameServer(
            run_id=run_id, host=host, port=port, working_directory=working_dir
        )
        ns_host, ns_port = ns.start()

        # Start local worker
        w = PyTorchWorker(
            run_id=run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            timeout=120,
        )
        w.run(background=True)

        # Run an optimizer
        bohb = BOHB(
            configspace=w.get_configspace(),
            run_id=run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=min_budget,
            max_budget=n_bohb_iterations,
        )

        result = bohb.run(n_iterations=1)
        inc_id = result.get_incumbent_id()

        end = result.get_runs_by_id(inc_id)[-1]["info"]
        res = [end["test_accuracy"], end["train_accuracy"], end["valid_accuracy"]]
        err_msg = "Worker compute function not implemented correctly."
        np.testing.assert_allclose(res, worker_end_res, atol=1e-5, err_msg=err_msg)

    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()


if __name__ == "__main__":
    test_pytorch_worker()
    print("Test complete.")
