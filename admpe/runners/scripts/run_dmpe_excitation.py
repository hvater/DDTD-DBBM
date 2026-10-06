import numpy as np

from admpe.runners.excitation_experiment import run_excitation_experiment


if __name__ == "__main__":

    for system_name in ["fluid_tank", "pendulum", "cart_pole"]:
        for excitation_objective in ["dmpe_static_dynamics", "dmpe_uniform"]:
            run_excitation_experiment(
                system_name=system_name,
                excitation_objective=excitation_objective,
                gpu_id=2,
                exp_name="run1",
                seeds=np.arange(0, 10, 1).tolist(),
            )
