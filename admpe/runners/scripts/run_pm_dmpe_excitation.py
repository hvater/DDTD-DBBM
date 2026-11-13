import os
import numpy as np

import optax

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS
from admpe.runners.excitation_experiment import run_excitation_experiment

if __name__ == "__main__":

    for system_name in SUPPORTED_SYSTEMS:

        for excitation_objective in [
            "dmpe_uniform",
            "dmpe_static_dynamics",
            "dmpe_static_curv",
            "dmpe_static_mixture_dynamics",
            "dmpe_static_mixture_curv",
        ]:
            run_excitation_experiment(
                system_name=system_name,
                excitation_objective=excitation_objective,
                gpu_id=0,
                exp_name="initial_submission",
                model_type="perfect",
                n_time_steps=7_000,
                seeds=np.arange(30, 35, 1).tolist(),
                noise_variance=0.0,
                constrain_to_ses=True,
                alg_overrides={
                    "consider_action_distribution": False,
                    "start_optimizing": -1,
                    "clip_action": True,
                    "excitation_optimizer": optax.adabelief(1e-2),
                    "n_opt_steps": 50,
                },
            )
