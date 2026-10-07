"""Metrics for state-transition and trajectory rollout prediction."""

import jax.numpy as jnp


def prediction_metrics(reference, prediction):
    """Inputs have shape (..., n_features).

    Reduction is performed over all dimensions except n_features,
    which is also reduced over for some of the metrics.

    Usual use will have input shape (n_samples, n_features).
    """

    reference = jnp.asarray(reference)
    prediction = jnp.asarray(prediction)

    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes must match.")
    if reference.ndim < 2:
        raise ValueError("Expected at least (n_samples, n_features).")

    error = prediction - reference

    # reduction will take place over all axes except for the features
    reduction_axes = tuple(range(error.ndim - 1))

    squared_error = jnp.square(error)

    return {
        "rmse": jnp.sqrt(jnp.mean(squared_error)),
        "rmse_per_feature": jnp.sqrt(jnp.mean(squared_error, axis=reduction_axes)),
        "mae": jnp.mean(jnp.abs(error)),
        "mae_per_feature": jnp.mean(jnp.abs(error), axis=reduction_axes),
        "wce": jnp.max(jnp.abs(error)),
        "wce_per_feature": jnp.max(jnp.abs(error), axis=reduction_axes),
        "nonfinite_prediction_fraction": jnp.mean(~jnp.all(jnp.isfinite(prediction), axis=-1)),
    }


def rollout_metrics(reference, prediction, *, horizons=None):
    """Inputs have shape (n_trajectories, n_steps + 1, n_features)."""

    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes must match.")
    if reference.ndim != 3:
        raise ValueError("Expected input shape to be (n_trajectories, n_steps + 1, n_features).")
    if reference.shape[-2] < 2:
        raise ValueError("A rollout must contain at least one transition.")

    n_steps = reference.shape[1] - 1
    n_features = reference.shape[2]

    if horizons is not None:
        if any(h < 1 or h > n_steps for h in horizons):
            raise ValueError("Horizons must lie between 1 and n_steps.")

    future_reference = reference[:, 1:]
    future_prediction = prediction[:, 1:]

    results = prediction_metrics(future_reference, future_prediction)
    squared_error = jnp.square((future_prediction - future_reference))

    results["rmse_by_step"] = jnp.sqrt(jnp.mean(squared_error, axis=(0, 2)))
    results["rmse_per_trajectory"] = jnp.sqrt(jnp.mean(squared_error, axis=(1, 2)))
    results["nonfinite_trajectory_fraction"] = jnp.mean(~jnp.all(jnp.isfinite(future_prediction), axis=(1, 2)))

    results["horizons"] = {
        int(h): {
            "terminal": prediction_metrics(reference[:, h], prediction[:, h]),
            "averaged": prediction_metrics(reference[:, 1 : h + 1], prediction[:, 1 : h + 1]),
        }
        for h in horizons
    }
    return results
