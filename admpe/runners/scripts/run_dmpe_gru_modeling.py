import os
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from admpe.data_management import list_excitation_results, provide_excitation_results_overview
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

    # results = list_excitation_results(
    #     system_name=system_name,
    #     exp_name=exp_name,
    #     excitation_objective=[
    #         "dmpe_uniform",
    #         "dmpe_static_dynamics",
    #         "dmpe_static_curv",
    #         "dmpe_static_LOLA_ls",
    #         "dmpe_static_mixture_dynamics",
    #         "dmpe_static_mixture_curv",
    #     ],
    #     model_type="perfect",
    #     status="completed",
    # )
    # dataset_paths = [result["path"] for result in results]

    # run_modeling_experiment(
    #     system_name=system_name,
    #     dataset_paths=dataset_paths,
    #     model_type="gru",
    #     gpu_id=2,
    #     exp_name=exp_name,
    #     seeds=np.arange(40, 43, 1).tolist(),
    #     perform_validation=True,
    #     validation_every=250,
    #     selection_metric="rollout_training_horizon_mse",
    #     model_overrides={"hidden_size": 128},
    #     model_trainer_overrides={
    #         "n_iters": 50_000,
    #         "sequence_length": 50,
    #     },
    # )

    exp_name = "submission_nov"

    for system_name in ["pmsm_nominal"]:

        results = list_excitation_results(
            system_name=system_name,
            exp_name=exp_name,
            excitation_objective=[
                "dmpe_uniform",
                "dmpe_static_dynamics",
                "dmpe_static_curv",
                "dmpe_static_LOLA_ls",
                "dmpe_static_mixture_dynamics",
                "dmpe_static_mixture_curv",
            ],
            model_type="perfect",
            status="completed",
        )
        dataset_paths = [result["path"] for result in results]

        run_modeling_experiment(
            system_name=system_name,
            dataset_paths=dataset_paths,
            model_type="gru",
            gpu_id=2,
            exp_name=exp_name,
            seeds=np.arange(40, 43, 1).tolist(),
            perform_validation=True,
            validation_every=250,
            selection_metric="rollout_training_horizon_mse",
            model_overrides={"hidden_size": 128},
            model_trainer_overrides={
                "n_iters": 15_000,
                "sequence_length": 100,
            },
        )
