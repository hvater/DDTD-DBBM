import pathlib
import json
from collections import defaultdict
import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results, list_modeling_results
from admpe.runners.modeling_experiment import run_modeling_experiment

if __name__ == "__main__":

    # exp_name = "state_only_pm_no_noise"

    # for system_name in ["pendulum"]:  # ["fluid_tank", "pendulum", "cart_pole"]:

    #     results = list_excitation_results(
    #         system_name=system_name,
    #         exp_name=exp_name,
    #         excitation_objective=["dmpe_uniform", "dmpe_static_curv", "dmpe_static_dynamics", "dmpe_static_LOLA_ls"],
    #         model_type="perfect",
    #         status="completed",
    #     )
    #     dataset_paths = [result["path"] for result in results]

    #     run_modeling_experiment(
    #         system_name=system_name,
    #         dataset_paths=dataset_paths,
    #         model_type="gru",
    #         gpu_id=2,
    #         exp_name="gru_100seq_validation",
    #         seeds=[10, 11],  # np.arange(10, 11, 1).tolist(),
    #         perform_validation=True,
    #         validation_every=250,
    #         selection_metric="rollout_training_horizon_mse",
    #         model_overrides={"hidden_size": 128},
    #         model_trainer_overrides={
    #             "n_iters": 15_000,
    #             "sequence_length": 100,  # maybe 50? currently running w/ 50
    #         },
    #     )

    # exp_name = "submission_nov"

    # for system_name in ["fluid_tank"]:  # ["fluid_tank", "pendulum", "cart_pole"]:

    #     results = list_excitation_results(
    #         system_name=system_name,
    #         exp_name=exp_name,
    #         excitation_objective=[
    #             "dmpe_uniform",
    #             "dmpe_static_dynamics",
    #             "dmpe_static_curv",
    #             "dmpe_static_LOLA_ls",
    #             "dmpe_static_mixture_dynamics",
    #             "dmpe_static_mixture_curv",
    #         ],
    #         model_type="perfect",
    #         status="completed",
    #     )
    #     dataset_paths = [result["path"] for result in results]

    #     run_modeling_experiment(
    #         system_name=system_name,
    #         dataset_paths=dataset_paths,
    #         model_type="gru",
    #         gpu_id=-1,
    #         exp_name="submission_nov",
    #         seeds=np.arange(40, 43, 1).tolist(),
    #         perform_validation=True,
    #         validation_every=250,
    #         selection_metric="rollout_training_horizon_mse",
    #         model_overrides={"hidden_size": 32},
    #         model_trainer_overrides={
    #             "n_iters": 5_000,
    #             "sequence_length": 100,  # maybe 50? currently running w/ 50
    #         },
    #     )

    # exp_name = "submission_nov"
    # system_name = "cart_pole"

    # modeling_seeds = np.arange(40, 43, 1).tolist()

    # completed = set()

    # for result in list_modeling_results(
    #     system_name=system_name,
    #     model_type="gru",
    #     exp_name=exp_name,
    #     status="completed",
    # ):
    #     path = pathlib.Path(result["path"])
    #     if not (path / "model.eqx").is_file():
    #         continue

    #     metadata = json.loads((path / "experiment.json").read_text())

    #     # Match dataset IDs even if their absolute paths have changed.
    #     saved_path = str(metadata["dataset_path"]).removeprefix("PosixPath('").removesuffix("')")
    #     completed.add((pathlib.Path(saved_path).name, int(metadata["seed"])))

    # # Batch datasets that need the same modeling seeds.
    # missing_groups = defaultdict(list)

    # for result in list_excitation_results(
    #     system_name=system_name,
    #     model_type="perfect",
    #     exp_name=exp_name,
    #     status="completed",
    # ):
    #     dataset_path = pathlib.Path(result["path"])
    #     missing_seeds = tuple(seed for seed in modeling_seeds if (dataset_path.name, seed) not in completed)
    #     if missing_seeds:
    #         missing_groups[missing_seeds].append(dataset_path)

    # n_missing = sum(len(seeds) * len(paths) for seeds, paths in missing_groups.items())
    # print(f"Scheduling {n_missing} missing modeling runs.")

    # for seeds, dataset_paths in missing_groups.items():
    #     for path in dataset_paths:
    #         print(f"{path.name}: missing modeling seeds {seeds}")

    #     run_modeling_experiment(
    #         system_name=system_name,
    #         dataset_paths=dataset_paths,
    #         model_type="gru",
    #         gpu_id=2,
    #         exp_name=exp_name,
    #         seeds=list(seeds),
    #         perform_validation=True,
    #         validation_every=250,
    #         selection_metric="rollout_training_horizon_mse",
    #         model_overrides={"hidden_size": 128},
    #         model_trainer_overrides={
    #             "n_iters": 50_000,
    #             "sequence_length": 50,
    #         },
    #     )

    # exp_name = "submission_nov"

    # for system_name in ["pmsm_nominal"]:

    #     results = list_excitation_results(
    #         system_name=system_name,
    #         exp_name=exp_name,
    #         excitation_objective=[
    #             "dmpe_uniform",
    #             "dmpe_static_dynamics",
    #             "dmpe_static_curv",
    #             "dmpe_static_LOLA_ls",
    #             "dmpe_static_mixture_dynamics",
    #             "dmpe_static_mixture_curv",
    #         ],
    #         model_type="perfect",
    #         status="completed",
    #     )
    #     dataset_paths = [result["path"] for result in results]

    #     run_modeling_experiment(
    #         system_name=system_name,
    #         dataset_paths=dataset_paths,
    #         model_type="gru",
    #         gpu_id=1,
    #         exp_name=exp_name,
    #         seeds=np.arange(40, 43, 1).tolist(),
    #         perform_validation=True,
    #         validation_every=250,
    #         selection_metric="rollout_training_horizon_mse",
    #         model_overrides={"hidden_size": 128},
    #         model_trainer_overrides={
    #             "n_iters": 50_000,
    #             "sequence_length": 50,
    #         },
    #     )

    # exp_name = "submission_nov"
    # system_name = "pmsm_nominal"

    # modeling_seeds = np.arange(40, 43, 1).tolist()

    # completed = set()

    # for result in list_modeling_results(
    #     system_name=system_name,
    #     model_type="gru",
    #     exp_name=exp_name,
    #     status="completed",
    # ):
    #     path = pathlib.Path(result["path"])
    #     if not (path / "model.eqx").is_file():
    #         continue

    #     metadata = json.loads((path / "experiment.json").read_text())

    #     # Match dataset IDs even if their absolute paths have changed.
    #     saved_path = str(metadata["dataset_path"]).removeprefix("PosixPath('").removesuffix("')")
    #     completed.add((pathlib.Path(saved_path).name, int(metadata["seed"])))

    # # Batch datasets that need the same modeling seeds.
    # missing_groups = defaultdict(list)

    # for result in list_excitation_results(
    #     system_name=system_name,
    #     model_type="perfect",
    #     exp_name=exp_name,
    #     status="completed",
    # ):
    #     dataset_path = pathlib.Path(result["path"])
    #     missing_seeds = tuple(seed for seed in modeling_seeds if (dataset_path.name, seed) not in completed)
    #     if missing_seeds:
    #         missing_groups[missing_seeds].append(dataset_path)

    # n_missing = sum(len(seeds) * len(paths) for seeds, paths in missing_groups.items())
    # print(f"Scheduling {n_missing} missing modeling runs.")

    # for seeds, dataset_paths in missing_groups.items():
    #     for path in dataset_paths:
    #         print(f"{path.name}: missing modeling seeds {seeds}")

    #     run_modeling_experiment(
    #         system_name=system_name,
    #         dataset_paths=dataset_paths,
    #         model_type="gru",
    #         gpu_id=1,
    #         exp_name=exp_name,
    #         seeds=list(seeds),
    #         perform_validation=True,
    #         validation_every=250,
    #         selection_metric="rollout_training_horizon_mse",
    #         model_overrides={"hidden_size": 128},
    #         model_trainer_overrides={
    #             "n_iters": 50_000,
    #             "sequence_length": 50,
    #         },
    #     )

    raise NotImplementedError()
