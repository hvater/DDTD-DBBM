import numpy as np

from admpe.data_management import list_excitation_results
from admpe.runners.modeling_experiment import run_modeling_experiment


if __name__ == "__main__":

    exp_name = "state_only_run1"

    for system_name in ["fluid_tank", "pendulum", "cart_pole"]:

        results = list_excitation_results(system_name=system_name, exp_name=exp_name)
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="neural_euler",
            gpu_id=1,
            exp_name=exp_name,
            seeds=np.arange(0, 5, 1).tolist(),
        )
