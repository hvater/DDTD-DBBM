import os
import numpy as np

from admpe.data_management import list_excitation_results
from admpe.runners.modeling_experiment import run_modeling_experiment

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

if __name__ == "__main__":

    exp_name = "state_only_run2"

    for system_name in ["fluid_tank", "pendulum", "cart_pole"]:

        results = list_excitation_results(
            system_name=system_name,
            exp_name=exp_name,
            excitation_objective=None,
        )
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="neural_euler",
            gpu_id=0,
            exp_name=exp_name,
            seeds=np.arange(0, 5, 1).tolist(),
        )
