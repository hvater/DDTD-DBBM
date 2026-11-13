import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


from admpe.evaluation.test_trajectories.generate_test_trajectories import (
    run_testing_dataset_experiment,
)
from admpe.data_management import VALIDATION_TRAJECTORIES_ROOT

##############################################################

use_safely_excitable_set = True

if use_safely_excitable_set:
    output_root = VALIDATION_TRAJECTORIES_ROOT / "safely_excitable_trajectories"
else:
    output_root = VALIDATION_TRAJECTORIES_ROOT / "box_constrained_trajectories"
output_root.mkdir(parents=True, exist_ok=True)

##############################################################

for system_name in ["fluid_tank", "pendulum", "cart_pole"]:
    for signal_name in ["aprbs", "multisine"]:

        run_testing_dataset_experiment(
            system_name=system_name,
            signal_name=signal_name,
            seed=151251,
            exp_name="ses_val_data",
            use_f64=False,
            gpu_id=0,
            n_sequences=5000,
            n_time_steps=250 if system_name == "cart_pole" else 500,
            batch_size=1024,
            use_safely_excitable_set=use_safely_excitable_set,
            noise_variance=0.0,
            max_attempts=int(1e7),
            output_root=output_root,
        )

    run_testing_dataset_experiment(
        system_name=system_name,
        signal_name="ocp",
        seed=42424,
        exp_name="ses_val_data",
        use_f64=False,
        gpu_id=0,
        n_sequences=655 if system_name == "cart_pole" else 5000,
        n_time_steps=500,  # 100 if system_name == "cart_pole" else 500,
        batch_size=2048,
        use_safely_excitable_set=use_safely_excitable_set,
        noise_variance=0.0,
        max_attempts=int(1e6),
        output_root=output_root,
    )


system_name = "pmsm_nominal"

run_testing_dataset_experiment(
    system_name=system_name,
    signal_name="pi",
    seed=42424,
    exp_name="ses_val_pi",
    use_f64=False,
    gpu_id=0,
    n_sequences=15000,
    n_time_steps=1001,  # 100 if system_name == "cart_pole" else 500,
    batch_size=2048,
    use_safely_excitable_set=use_safely_excitable_set,
    noise_variance=0.0,
    max_attempts=int(1e6),
    output_root=output_root,
)
