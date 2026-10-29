import equinox as eqx
import jax.numpy as jnp


class PersistenceModel(eqx.Module):
    """Simple baseline model that just returns the momentary observation.
    Used to evaluate a lower bound on the one step prediction performance.
    """

    def step(self, obs, action, tau):
        return obs

    def __call__(self, init_obs, actions, tau):
        return jnp.broadcast_to(
            init_obs,
            (actions.shape[0] + 1, init_obs.shape[-1]),
        )
