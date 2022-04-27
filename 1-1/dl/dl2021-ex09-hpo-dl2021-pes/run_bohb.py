import os
from lib.worker import PyTorchWorker
from lib.utilities import save_result
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB


def run_bohb(
        host: str,
        port: int,
        run_id: str,
        n_bohb_iterations: int,
        working_dir: str,
        min_budget: int,
        max_budget: int) -> None:
    """Run BOHB.

    Returns:
        None

    """
    try:
        # Start a nameserver #####
        ns = hpns.NameServer(run_id=run_id, host=host, port=port,
                             working_directory=working_dir)
        ns_host, ns_port = ns.start()

        # Start local worker
        w = PyTorchWorker(run_id=run_id, host=host, nameserver=ns_host,
                          nameserver_port=ns_port, timeout=120)
        w.run(background=True)

        # Run an optimizer
        bohb = BOHB(configspace=w.get_configspace(),
                    run_id=run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    min_budget=min_budget, max_budget=max_budget)

        result = bohb.run(n_iterations=n_bohb_iterations)
        save_result('bohb_result', result)
    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()


if __name__ == '__main__':
    # minimum budget that BOHB uses
    min_budget = 1
    # largest budget BOHB will use
    max_budget = 9
    working_dir = os.curdir
    host = "localhost"
    port = 0
    run_id = 'bohb_run_1'
    n_bohb_iterations = 12
    run_bohb(
        host,
        port,
        run_id,
        n_bohb_iterations,
        working_dir,
        min_budget,
        max_budget)
