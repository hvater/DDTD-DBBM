import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results
from admpe.runners.modeling_experiment import run_modeling_experiment

from admpe.utils.helper import missing_batches

if __name__ == "__main__":

    exp_name = "submission_nov"

    for system_name in [
        "cart_pole",
        # "pmsm_nominal",
        # "fluid_tank",
        # "pendulum",
    ]:

        groups = missing_batches(
            system_name,
            "neural_euler_large_1",
            "submission_nov_1step",
            list(range(50, 55)),
        )

        for seed, runs in groups.items():
            print("seed: ", seed, "\n", "n_runs:", len(runs), "\n")

        # results = list_excitation_results(
        #     system_name=system_name,
        #     exp_name=exp_name,
        #     excitation_objective=None,
        #     model_type="perfect",
        # )
        # dataset_paths = [result["path"] for result in results]

        print("Missing fits:", sum(len(seeds) * len(paths) for seeds, paths in groups.items()))

        for seeds, dataset_paths in groups.items():
            run_modeling_experiment(
                system_name=system_name,
                dataset_paths=dataset_paths,
                model_type="neural_euler_large_1",
                gpu_id=0,
                exp_name="submission_nov_1step",
                seeds=list(seeds),
                perform_validation=True,
                validation_every=250,
                selection_metric="rollout_training_horizon_mse",
                model_trainer_overrides={
                    "n_iters": 15_000,
                },
            )
