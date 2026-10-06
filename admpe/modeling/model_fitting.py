from tqdm import tqdm

import jax.numpy as jnp


def fit(
    train_data,
    model_type,
    model,
    model_trainer,
    opt_state,
    loader_key,
    params,
):

    if model_type.startswith("gp_"):
        raise NotImplementedError()

    observations = train_data["observations"]
    actions = train_data["actions"]

    for iteration in tqdm(range(params["n_iters"])):
        model, opt_state, loader_key = model_trainer.fit(
            model=model,
            k=jnp.array([observations.shape[0]]),
            observations=observations,
            actions=actions,
            opt_state=opt_state,
            loader_key=loader_key,
        )

    # TODO: actual evaluation loss and proper testing for best_model performance
    best_model = model

    return best_model
