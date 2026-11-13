import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results
from admpe.excitation.excitation_experiment_setup import SUPPORTED_SYSTEMS
from admpe.runners.modeling_experiment import run_modeling_experiment

from admpe.utils.helper import missing_batches

if __name__ == "__main__":
    exp_name = "initial_submission"

    for system_name in SUPPORTED_SYSTEMS:

        results = list_excitation_results(
            system_name=system_name, exp_name=exp_name, excitation_objective=None, model_type="perfect"
        )
        dataset_paths = [result["path"] for result in results]
        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="gp_transition",
            seeds=[6876678],
            gpu_id=-1,
            use_f64=True,
            exp_name="initial_submission",
            perform_validation=False,
            model_trainer_overrides={
                "max_iters": 40,
            },
        )
