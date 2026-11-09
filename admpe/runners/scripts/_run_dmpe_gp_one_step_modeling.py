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
    ]:  # [ "cart_pole"]:

        groups = missing_batches(system_name, "gp_transition", "submission_nov", [6876678])

        for seed, runs in groups.items():
            print("seed: ", seed, "\n", "n_runs:", len(runs), "\n")

        # results = list_excitation_results(
        #     system_name=system_name, exp_name=exp_name, excitation_objective=None, model_type="perfect"
        # )
        # dataset_paths = [result["path"] for result in results]

        print("Missing fits:", sum(len(seeds) * len(paths) for seeds, paths in groups.items()))

        for seeds, dataset_paths in groups.items():

            run_modeling_experiment(
                system_name=system_name,
                dataset_paths=dataset_paths,
                model_type="gp_transition",
                seeds=list(seeds),
                gpu_id=-1,
                use_f64=True,
                exp_name="submission_nov",  # "state_only_pm_no_noise_one_step",
                perform_validation=False,
                model_trainer_overrides={
                    "max_iters": 40,
                },
            )
