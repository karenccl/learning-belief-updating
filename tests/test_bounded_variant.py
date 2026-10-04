"""Post-hoc sensitivity variant: beta <= 20 (scaled logistic) and Bayesian h in (0, 0.5)."""
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import load_config
from src.fit import boundary_flags, fit_arrays
from src.models import (
    BETA_MAX, MODEL_PARAMS, PRIMARY_MODELS, negative_log_likelihood, reference_nll, transform, untransform,
)
from src.simulate import simulate_model
from src.task import environment_for_condition

TASK = load_config()["task"]


@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_bounded_transforms_respect_bounds(model):
    names = MODEL_PARAMS[model]
    for raw in (-30.0, -3.0, 0.0, 3.0, 30.0):
        p = dict(zip(names, np.asarray(transform(model, jnp.full(len(names), raw), "bounded"))))
        assert 1e-4 <= p["beta"] <= BETA_MAX
        if "h" in p:
            assert 0 < p["h"] < 0.5 or (raw >= 30 and p["h"] <= 0.5)


def test_bounded_roundtrip_and_locked_default_unchanged():
    params = {"h": 0.2, "beta": 6.0}
    theta = untransform("bayes_cp", params, "bounded")
    np.testing.assert_allclose(np.asarray(transform("bayes_cp", jnp.asarray(theta), "bounded")), [0.2, 6.0])
    # locked transform is still softplus / logistic
    np.testing.assert_allclose(np.asarray(transform("bayes_cp", jnp.zeros(2))), [0.5, np.log(2) + 1e-4])


def test_bounded_likelihood_matches_reference_at_same_natural_parameters():
    env = environment_for_condition("volatile", 160, TASK)
    df = simulate_model("bayes_cp", {"h": 0.1, "beta": 5.0}, env, seed=4)
    params = {"h": 0.15, "beta": 12.0}
    v = negative_log_likelihood(untransform("bayes_cp", params, "bounded"), df.choice, df.reward,
                                np.ones(160), "bayes_cp", "bounded")
    assert v == pytest.approx(reference_nll("bayes_cp", params, df.choice, df.reward), abs=1e-8)


def test_bounded_fit_stays_inside_bounds_where_locked_runs_away():
    # A near-random agent: the locked fit tends to run away; the bounded fit cannot.
    env = environment_for_condition("moderate", 80, TASK)
    df = simulate_model("fixed_rl", {"alpha": 0.3, "beta": 0.6}, env, seed=31)
    fit = fit_arrays("bayes_cp", df.choice, df.reward, n_starts=10, rng=np.random.default_rng(0),
                     variant="bounded")
    assert fit["variant"] == "bounded" and not fit["failed"]
    assert fit["params"]["beta"] <= BETA_MAX and fit["params"]["h"] <= 0.5


def test_bounded_boundary_flags():
    assert boundary_flags({"h": 0.499, "beta": 19.9}, "bounded") == {"h": True, "beta": True}
    assert boundary_flags({"h": 0.2, "beta": 19.0}, "bounded") == {"h": False, "beta": False}
    assert boundary_flags({"h": 0.2, "beta": 19.9}) == {"h": False, "beta": False}  # locked: beta > 20 only
