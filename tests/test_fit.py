import math

import jax.numpy as jnp
import numpy as np
import pytest

from src.config import load_config
from src.fit import boundary_flags, fit_model
from src.models import (
    ALL_MODELS, MODEL_PARAMS, negative_log_likelihood, nll_and_grad_fn, reference_nll, transform,
    untransform,
)
from src.simulate import simulate_model
from src.task import environment_for_condition

TASK = load_config()["task"]
EXAMPLE_PARAMS = {
    "random": {},
    "perseveration": {"bias": -0.4, "rho": 1.2},
    "fixed_rl": {"alpha": 0.3, "beta": 4.0},
    "asymmetric_rl": {"alpha_pos": 0.5, "alpha_neg": 0.15, "beta": 3.0},
    "hybrid_ph": {"eta": 0.4, "kappa": 0.7, "beta": 5.0},
    "bayes_cp": {"h": 0.1, "beta": 6.0},
}


def _log_sigmoid(x):
    return -math.log1p(math.exp(-x))


def test_hand_calculated_three_trial_fixed_rl_likelihood():
    # alpha = 0.3, beta = 4, choices 0,0,1, rewards 1,0,1 (hand calculation in docs/notes/day1.md)
    # t1: Q=(.5,.5) logit 0            -> log P(a=0) = log .5;         Q0 <- .65
    # t2: Q=(.65,.5) logit 4(.5-.65)   -> log P(a=0) = log s(0.6);     Q0 <- .455
    # t3: Q=(.455,.5) logit 4(.045)    -> log P(a=1) = log s(0.18)
    expected = -(math.log(0.5) + _log_sigmoid(0.6) + _log_sigmoid(0.18))
    assert expected == pytest.approx(1.737827, abs=1e-6)
    choices, rewards = np.array([0, 0, 1]), np.array([1.0, 0.0, 1.0])
    theta = untransform("fixed_rl", {"alpha": 0.3, "beta": 4.0})
    assert negative_log_likelihood(theta, choices, rewards, np.ones(3), "fixed_rl") == pytest.approx(
        expected, abs=1e-10)
    assert reference_nll("fixed_rl", {"alpha": 0.3, "beta": 4.0}, choices, rewards) == pytest.approx(
        expected, abs=1e-12)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_likelihood_finite_for_valid_parameters(model):
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model(model, EXAMPLE_PARAMS[model], env, seed=8)
    rng = np.random.default_rng(1)
    for _ in range(5):
        theta = rng.normal(0, 3, size=len(MODEL_PARAMS[model]))
        v = negative_log_likelihood(theta, df.choice, df.reward, np.ones(160), model)
        assert np.isfinite(v) and v > 0


@pytest.mark.parametrize("model", ALL_MODELS)
def test_jax_matches_reference_within_1e_8(model):
    rng = np.random.default_rng(42)
    for cond in ("stable", "volatile"):
        env = environment_for_condition(cond, 160, TASK)
        df = simulate_model(model, EXAMPLE_PARAMS[model], env, seed=int(rng.integers(1e6)))
        mask = (rng.random(160) < 0.7).astype(float)
        for _ in range(4):
            theta = rng.normal(0, 1.5, size=len(MODEL_PARAMS[model]))
            params = dict(zip(MODEL_PARAMS[model], np.asarray(transform(model, jnp.asarray(theta)))))
            ref = reference_nll(model, params, df.choice, df.reward, mask)
            jx = negative_log_likelihood(theta, df.choice, df.reward, mask, model)
            assert jx == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("model", [m for m in ALL_MODELS if MODEL_PARAMS[m]])
def test_jax_gradient_matches_finite_differences(model):
    env = environment_for_condition("volatile", 160, TASK)
    df = simulate_model(model, EXAMPLE_PARAMS[model], env, seed=3)
    c, r, m = df.choice.to_numpy(), df.reward.to_numpy(float), np.ones(160)
    vg = nll_and_grad_fn(model)
    rng = np.random.default_rng(0)
    for _ in range(3):
        theta = rng.normal(0, 1, size=len(MODEL_PARAMS[model]))
        _, g = vg(jnp.asarray(theta), jnp.asarray(c), jnp.asarray(r), jnp.asarray(m))
        eps = 1e-6
        fd = np.array([
            (negative_log_likelihood(theta + eps * e, c, r, m, model)
             - negative_log_likelihood(theta - eps * e, c, r, m, model)) / (2 * eps)
            for e in np.eye(len(theta))
        ])
        np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-5, atol=1e-6)


def test_invalid_raw_parameters_cannot_produce_invalid_transformed_parameters():
    for model in ("fixed_rl", "asymmetric_rl", "hybrid_ph", "bayes_cp", "perseveration"):
        names = MODEL_PARAMS[model]
        for raw in (-30.0, -5.0, 0.0, 5.0, 30.0):
            p = dict(zip(names, np.asarray(transform(model, jnp.full(len(names), raw)))))
            for n, v in p.items():
                assert np.isfinite(v)
                if n in ("alpha", "alpha_pos", "alpha_neg", "eta", "kappa", "h"):
                    assert 0 < v < 1
                if n == "beta":
                    assert v >= 1e-4
                if n == "rho":
                    assert -5 <= v <= 5  # tanh saturates to exactly +-5 at |raw| = 30
                    if abs(raw) <= 5:
                        assert -5 < v < 5
        for raw in (-1e6, 1e6):  # extreme values saturate but stay valid and finite
            p = np.asarray(transform(model, jnp.full(len(names), raw)))
            assert np.isfinite(p).all()


def test_transform_roundtrip():
    for model, params in EXAMPLE_PARAMS.items():
        theta = untransform(model, params)
        back = np.asarray(transform(model, jnp.asarray(theta)))
        np.testing.assert_allclose(back, [params[n] for n in MODEL_PARAMS[model]], rtol=1e-10)


def test_likelihood_lower_near_truth_than_at_implausible_parameters():
    env = environment_for_condition("moderate", 320, TASK)
    truth = {"alpha": 0.3, "beta": 5.0}
    df = simulate_model("fixed_rl", truth, env, seed=12)
    m = np.ones(320)
    at_truth = negative_log_likelihood(untransform("fixed_rl", truth), df.choice, df.reward, m, "fixed_rl")
    for bad in ({"alpha": 0.99, "beta": 0.2}, {"alpha": 0.01, "beta": 5.0}, {"alpha": 0.3, "beta": 30.0}):
        v = negative_log_likelihood(untransform("fixed_rl", bad), df.choice, df.reward, m, "fixed_rl")
        assert at_truth < v


def test_fit_model_returns_finite_diagnostics_and_recovers_parameters():
    env = environment_for_condition("moderate", 320, TASK)
    truth = {"alpha": 0.3, "beta": 5.0}
    df = simulate_model("fixed_rl", truth, env, seed=99)
    fit = fit_model(df, "fixed_rl", n_starts=10, seed=1)
    assert not fit["failed"] and fit["converged"]
    assert np.isfinite(fit["loss"]) and len(fit["starts"]) == 10
    assert {"loss", "grad_norm", "status", "nit", "params", "converged"} <= set(fit["starts"][0])
    assert abs(fit["params"]["alpha"] - 0.3) < 0.15
    assert abs(fit["params"]["beta"] - 5.0) < 2.0
    assert not fit["near_random"]
    assert 0 < fit["prop_starts_within_0.1"] <= 1


def test_near_random_flag_for_random_behaviour():
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model("random", {}, env, seed=5)
    fit = fit_model(df, "fixed_rl", n_starts=10, seed=2)
    assert not fit["failed"]
    assert fit["near_random"]  # flagged ...
    assert np.isfinite(fit["params"]["alpha"])  # ... but kept


def test_boundary_flags():
    assert boundary_flags({"alpha": 0.005, "beta": 3.0}) == {"alpha": True, "beta": False}
    assert boundary_flags({"alpha": 0.5, "beta": 25.0}) == {"alpha": False, "beta": True}
    assert boundary_flags({"alpha": 0.995, "beta": 7.9}) == {"alpha": True, "beta": False}
    assert boundary_flags({"bias": 3.0, "rho": -4.95}) == {"bias": False, "rho": True}
