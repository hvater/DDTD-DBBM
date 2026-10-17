import numpy as np

from admpe.runners.excitation_experiment import run_excitation_experiment


if __name__ == "__main__":

    for system_name in ["fluid_tank", "pendulum", "cart_pole"]:
        for excitation_objective in ["dmpe_static_dynamics", "dmpe_uniform", "dmpe_static_mixture_dynamics"]:
            run_excitation_experiment(
                system_name=system_name,
                excitation_objective=excitation_objective,
                gpu_id=2,
                exp_name="state_only_run1",
                seeds=np.arange(0, 10, 1).tolist(),
                alg_overrides={"consider_action_distribution": False},
            )
