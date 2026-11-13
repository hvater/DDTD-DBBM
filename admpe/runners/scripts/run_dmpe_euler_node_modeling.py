import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results
from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS
from admpe.runners.modeling_experiment import run_modeling_experiment

if __name__ == "__main__":
    exp_name = "initial_submission"

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
            model_type="neural_euler_large_1",
            gpu_id=0,
            exp_name="initial_submission",
            seeds=list(range(50, 55)),
            perform_validation=True,
            validation_every=250,
            selection_metric="rollout_training_horizon_mse",
            model_trainer_overrides={
                "n_iters": 15_000,
            },
        )
