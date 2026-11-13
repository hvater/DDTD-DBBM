import pathlib
import json
from collections import defaultdict
import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results, list_modeling_results
from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS
from admpe.runners.modeling_experiment import run_modeling_experiment

if __name__ == "__main__":
    exp_name = "initial_submission"

    hidden_size_per_system = {
        "fluid_tank": 32,
        "pendulum": 128,
        "cart_pole": 128,
        "pmsm_nominal": 128,
    }

    n_iters_per_system = {
        "fluid_tank": 5_000,
        "pendulum": 15_000,
        "cart_pole": 50_000,
        "pmsm_nominal": 50_000,
    }

    sequence_length_per_system = {
        "fluid_tank": 100,
        "pendulum": 100,
        "cart_pole": 50,
        "pmsm_nominal": 50,
    }

    for system_name in SUPPORTED_SYSTEMS:
        results = list_excitation_results(
            system_name=system_name,
            exp_name=exp_name,
            excitation_objective=None,
            model_type="perfect",
        )
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="gru",
            gpu_id=2,
            exp_name="gru_100seq_validation",
            seeds=np.arange(40, 43, 1).tolist(),
            perform_validation=True,
            validation_every=250,
            selection_metric="rollout_training_horizon_mse",
            model_overrides={"hidden_size": 128},
            model_trainer_overrides={
                "n_iters": n_iters_per_system[system_name],
                "sequence_length": sequence_length_per_system[system_name],
            },
        )
