from admpe.evaluation.test_trajectories.generate_test_trajectories import (
    SUPPORTED_SYSTEMS,
    SUPPORTED_SIGNALS,
    run_testing_dataset_experiment,
)

for system_name in SUPPORTED_SYSTEMS:
    # for signal_name in ["aprbs", "multisine"]:

    #     run_testing_dataset_experiment(
    #         system_name=system_name,
    #         signal_name=signal_name,
    #         seed=555,
    #         exp_name="run1",
    #         use_f64=False,
    #         gpu_id=0,
    #         n_sequences=5000,
    #         n_time_steps=100 if system_name == "cart_pole" else 500,
    #         batch_size=1024,
    #         noise_variance=0.0,
    #         max_attempts=int(1e6),
    #     )
    run_testing_dataset_experiment(
        system_name=system_name,
        signal_name="ocp",
        seed=555,
        exp_name="run1",
        use_f64=False,
        gpu_id=2,
        n_sequences=2000,
        n_time_steps=500,  # 100 if system_name == "cart_pole" else 500,
        batch_size=1024,
        noise_variance=0.0,
        max_attempts=int(1e6),
    )
