import os
import numpy as np


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.runners.excitation_experiment import run_excitation_experiment


if __name__ == "__main__":

    for system_name in ["fluid_tank", "pendulum", "cart_pole"]:  # [, "pendulum", "cart_pole"]:
        for excitation_objective in [
            "dmpe_static_dynamics",
            "dmpe_uniform",
            "dmpe_static_mixture_dynamics",
            "dmpe_static_curv",
            "dmpe_static_mixture_curv",
            "dmpe_static_LOLA_ls",
        ]:
            run_excitation_experiment(
                system_name=system_name,
                excitation_objective=excitation_objective,
                gpu_id=2,
                exp_name="state_only_run5",
                seeds=np.arange(0, 5, 1).tolist(),
                alg_overrides={"consider_action_distribution": False},
            )
