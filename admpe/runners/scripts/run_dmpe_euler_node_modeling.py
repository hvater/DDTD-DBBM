import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results
from admpe.runners.modeling_experiment import run_modeling_experiment


if __name__ == "__main__":

    exp_name = "state_only_pm_no_noise"

    for system_name in ["cart_pole"]:  # ["fluid_tank", "pendulum", "cart_pole"]:

        results = list_excitation_results(
            system_name=system_name,
            exp_name=exp_name,
            excitation_objective=None,
        )
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="neural_euler_large_250",
            gpu_id=0,
            exp_name="state_only_pm_no_noise_250_horizon",
            seeds=np.arange(0, 5, 1).tolist(),
            perform_validation=False,
        )
