from admpe.runners.excitation_experiment import run_excitation_experiment


if __name__ == "__main__":
    run_excitation_experiment(
        system_name="pendulum",
        excitation_objective="dmpe_uniform",
        gpu_id=0,
        exp_name="test",
        seeds=[42, 1033],
        alg_overrides=dict(bandwidth=0.17),
    )
