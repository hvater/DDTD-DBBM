import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results
from admpe.runners.modeling_experiment import run_modeling_experiment

if __name__ == "__main__":

    exp_name = "state_only_pm_no_noise"

    for system_name in ["pendulum"]:  # ["fluid_tank", "pendulum", "cart_pole"]:

        results = list_excitation_results(
            system_name=system_name,
            exp_name=exp_name,
            excitation_objective=None,
        )
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="gru",
            gpu_id=1,
            exp_name="state_only_gru",
            seeds=np.arange(10, 12, 1).tolist(),
            perform_validation=True,
            validation_every=1000,
            model_overrides={"hidden_size": 64},
            model_trainer_overrides={
                "n_iters": 50_000,
                "sequence_length": 30,  # maybe 50? currently running w/ 50
            },
        )
