# Dynamics-dependent Target Distributions in Dataset Generation for Nonlinear Dynamical Black-box System Models

This codebase accompanies the submission "Dynamics-dependent Target Distributions in Dataset Generation for Nonlinear Dynamical Black-box System Models" and was used in the generation of all presented results and experiments.

The two repositories

- https://github.com/ExcitingSystems/DMPE/tree/towards-v0.4
- https://github.com/ExcitingSystems/exciting-environments/tree/noisy-envs
  
are needed as dependencies using the specified branches (These will be properly incorporated in the future including detailed installation steps).


## System simulations:

The utilized simulators are implemented in the `exciting-environments` repository.

They are implemented as ordinary differential equations and solved using ODE solvers from https://docs.kidger.site/diffrax/.

- Fluid tank: 
    - The ODE implementation can be found [here](https://github.com/ExcitingSystems/exciting-environments/blob/noisy-envs/exciting_environments/fluid_tank/fluid_tank_env.py#L113).
    - The utilized system parameterization is specified [here](https://github.com/ExcitingSystems/DMPE/blob/main/dmpe/utils/env_utils/fluid_tank_utils.py#L22).
- Pendulum:
    - The ODE implementation can be found [here](https://github.com/ExcitingSystems/exciting-environments/blob/noisy-envs/exciting_environments/pendulum/pendulum_env.py#L154).
    - The utilized system parameterization is specified [here](https://github.com/ExcitingSystems/DMPE/blob/main/dmpe/utils/env_utils/pendulum_utils.py#L22).
- Cart-pole:
    - The ODE implementation can be found [here](https://github.com/ExcitingSystems/exciting-environments/blob/noisy-envs/exciting_environments/cart_pole/cart_pole_env.py#L172).
    - The utilized system parameterization is specified [here](https://github.com/ExcitingSystems/DMPE/blob/main/dmpe/utils/env_utils/cart_pole_utils.py#L22).
- PMSM:
    - The ODE implementation can be found [here](https://github.com/hvater/DDTD-DBBM/blob/main/admpe/pmsm.py#L159).
    - The utilized system parameterization is specified [here](https://github.com/hvater/DDTD-DBBM/blob/main/admpe/pmsm.py#L232).



## Excitation experiments:

The script for reproducing the excitation experiments can be found at 
`DDTD-DBBM/admpe/runners/scripts/run_pm_dmpe_excitation.py`.

Any parameterization that is not defined within this script uses the default parameterization provided at `DDTD-DBBM/admpe/excitation/excitation_parameterization.py`.


## Modeling experiments:

The scripts for the downstream modeling:

- GP: `DDTD-DBBM/admpe/runners/scripts/run_dmpe_gp_one_step_modeling.py`
- MLP: `DDTD-DBBM/admpe/runners/scripts/run_dmpe_euler_node_modeling.py`
- GRU: `DDTD/admpe/runners/scripts/run_dmpe_gru_modeling.py`


The scripts for the generation of 

- validation data:
- testing data:


## Reproduction of results:

- first run `DDTD-DBBM/admpe/runners/scripts/run_pm_dmpe_excitation.py` to generate all excitation datasets
- then run:
    - 
    - 
    -

Ensure that the `exp_name` used for the excitation datasets matches the loading `exp_name` in the modeling scripts