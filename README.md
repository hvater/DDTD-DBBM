# Dynamics-dependent Target Distributions in Dataset Generation for Nonlinear Dynamical Black-box System Models

This codebase accompanies the submission "Dynamics-dependent Target Distributions in Dataset Generation for Nonlinear Dynamical Black-box System Models" and was used in the generation of all presented results and experiments.

The two repositories

- https://github.com/ExcitingSystems/DMPE/tree/towards-v0.4
- https://github.com/ExcitingSystems/exciting-environments/tree/noisy-envs
  
are needed as dependencies using the specified branches (These will be properly incorporated in the future).


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

The excitation algorithm parameterization for each system is provided in the following:

- Fluid tank: 
    
- Pendulum:
    
- Cart-pole:
    
- PMSM:


## Modeling experiments:

The modeling algorithm parameterization for each system and each model type is provided in the following:

### GP:

- Fluid tank: 
    
- Pendulum:
    
- Cart-pole:
    
- PMSM:

### MLP:

- Fluid tank: 
    
- Pendulum:
    
- Cart-pole:
    
- PMSM:

### GRU:

- Fluid tank: 
    
- Pendulum:
    
- Cart-pole:
    
- PMSM: