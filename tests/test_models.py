import numpy as np
import pytest

from src.config import load_config
from src.models import (
    ALL_MODELS, MODEL_PARAMS, choice_logit, logps_fn, make_agent, reference_logps, sigmoid_np,
)
from src.simulate import simulate_model
from src.task import environment_for_condition

TASK = load_config()["task"]
EXAMPLE_PARAMS = {
    "random": {},
    "perseveration": {"bias": 0.3, "rho": 1.5},
    "fixed_rl": {"alpha": 0.3, "beta": 4.0},
    "asymmetric_rl": {"alpha_pos": 0.5, "alpha_neg": 0.15, "beta": 3.0},
    "hybrid_ph": {"eta": 0.4, "kappa": 0.7, "beta": 5.0},
    "bayes_cp": {"h": 0.1, "beta": 6.0},
}


def test_fixed_rl_value_increases_after_better_than_expected_reward():
    agent = make_agent("fixed_rl", {"alpha": 0.3, "beta": 4.0})
    agent.logit(0, 0.0)
    before = agent.Q[1]
    agent.update(1, 1)
    assert agent.Q[1] > before


def test_fixed_rl_value_decreases_after_no_reward_when_expected_positive():
    agent = make_agent("fixed_rl", {"alpha": 0.3, "beta": 4.0})
    agent.logit(0, 0.0)
    before = agent.Q[0]
    agent.update(0, 0)
    assert agent.Q[0] < before
    assert agent.Q[0] == pytest.approx(0.5 - 0.3 * 0.5)


def test_asymmetric_rl_learning_rate_branching():
    params = {"alpha_pos": 0.6, "alpha_neg": 0.1, "beta": 3.0}
    agent = make_agent("asymmetric_rl", params)
    agent.logit(0, 0.0)
    out = agent.update(0, 1)  # delta = +0.5
    assert out["alpha_t"] == 0.6 and agent.Q[0] == pytest.approx(0.5 + 0.6 * 0.5)
    agent.logit(1, -1.0)
    out = agent.update(1, 0)  # delta = -0.5
    assert out["alpha_t"] == 0.1 and agent.Q[1] == pytest.approx(0.5 - 0.1 * 0.5)
    assert agent.learning_rate(0.0) == 0.6  # delta = 0 counts as non-negative


def test_asymmetric_rl_jax_branching_matches_reference():
    # alpha_pos >> alpha_neg makes any branching error visible in the likelihood
    env = environment_for_condition("volatile", 160, TASK)
    params = {"alpha_pos": 0.7, "alpha_neg": 0.05, "beta": 5.0}
    df = simulate_model("asymmetric_rl", params, env, seed=11)
    ref = reference_logps("asymmetric_rl", params, df.choice, df.reward)
    jx = np.asarray(logps_fn("asymmetric_rl")(np.array([0.7, 0.05, 5.0]), df.choice.to_numpy(),
                                             df.reward.to_numpy(float)))
    np.testing.assert_allclose(jx, ref, atol=1e-10)


def test_adaptive_learner_learning_rate_strictly_between_0_and_1():
    env = environment_for_condition("volatile", 320, TASK)
    for eta, kappa in [(0.05, 0.1), (0.8, 0.9), (0.99, 0.99)]:
        df = simulate_model("hybrid_ph", {"eta": eta, "kappa": kappa, "beta": 4.0}, env, seed=5)
        assert ((df["alpha_t"] > 0) & (df["alpha_t"] < 1)).all()
        assert ((df["associability"] > 0) & (df["associability"] < 1)).all()


def test_hybrid_order_of_operations():
    agent = make_agent("hybrid_ph", {"eta": 0.5, "kappa": 0.8, "beta": 2.0})
    agent.logit(0, 0.0)
    out = agent.update(0, 1)  # delta = 0.5; alpha_t uses A_1 = 0.5
    assert out["alpha_t"] == pytest.approx(0.8 * 0.5)
    assert agent.Q[0] == pytest.approx(0.5 + 0.4 * 0.5)
    assert agent.A == pytest.approx(0.5 + 0.5 * (0.5 - 0.5))
    agent.logit(1, -1.0)
    out = agent.update(0, 0)  # delta = -0.7; A_2 still 0.5
    assert out["alpha_t"] == pytest.approx(0.4)
    assert agent.A == pytest.approx(0.5 + 0.5 * (0.7 - 0.5))


def test_hybrid_with_eta_zero_reproduces_fixed_rl():
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model("fixed_rl", {"alpha": 0.35, "beta": 4.0}, env, seed=21)
    kappa = 0.6
    hyb = reference_logps("hybrid_ph", {"eta": 0.0, "kappa": kappa, "beta": 4.0}, df.choice, df.reward)
    fix = reference_logps("fixed_rl", {"alpha": 0.5 * kappa, "beta": 4.0}, df.choice, df.reward)
    np.testing.assert_allclose(hyb, fix, atol=1e-12)
    hyb_jax = np.asarray(logps_fn("hybrid_ph")(np.array([0.0, kappa, 4.0]), df.choice.to_numpy(),
                                               df.reward.to_numpy(float)))
    np.testing.assert_allclose(hyb_jax, fix, atol=1e-10)


def test_bayesian_beliefs_are_valid_probabilities():
    env = environment_for_condition("volatile", 320, TASK)
    for h in (0.001, 0.1, 0.3, 0.9):
        df = simulate_model("bayes_cp", {"h": h, "beta": 5.0}, env, seed=2)
        b = df["belief_option_0"]
        assert ((b > 0) & (b < 1)).all()
        np.testing.assert_allclose(b + (1 - b), 1.0)
        # beliefs imply values that sum to pH + pL
        np.testing.assert_allclose(df["Q0"] + df["Q1"], 1.0)


def test_bayesian_two_trial_hand_calculation():
    h, beta = 0.1, 2.0
    agent = make_agent("bayes_cp", {"h": h, "beta": beta})
    # Trial 1: prior 0.5 is a fixed point of the transition
    x1 = agent.logit(0, 0.0)
    assert agent.b_pred == pytest.approx(0.5) and x1 == pytest.approx(0.0)
    agent.update(0, 1)  # l0 = 0.8, l1 = 0.2
    assert agent.b_post == pytest.approx(0.8)
    # Trial 2: transition 0.9*0.8 + 0.1*0.2 = 0.74; Q0 = 0.644, Q1 = 0.356
    x2 = agent.logit(1, -1.0)
    assert agent.b_pred == pytest.approx(0.74)
    q0, q1 = agent._q(agent.b_pred)
    assert (q0, q1) == (pytest.approx(0.644), pytest.approx(0.356))
    assert x2 == pytest.approx(beta * (0.356 - 0.644))
    agent.update(0, 0)  # l0 = 0.2, l1 = 0.8
    assert agent.b_post == pytest.approx(0.2 * 0.74 / (0.2 * 0.74 + 0.8 * 0.26))


def test_bayesian_deterministic_evidence_moves_belief_correctly():
    h = 0.05
    agent = make_agent("bayes_cp", {"h": h, "beta": 1.0})
    beliefs = []
    for t in range(30):  # option 1 always rewarded -> belief that 0 is best falls
        agent.logit(t, 0.0)
        agent.update(1, 1)
        beliefs.append(agent.b_post)
    assert np.all(np.diff(beliefs) <= 0) and beliefs[5] < beliefs[0]
    # The hazard keeps the predictive belief >= h * (1 - b), so the posterior converges
    # to the fixed point of b = 0.2 b' / (0.2 b' + 0.8 (1 - b')), b' = (1-h) b + h (1-b).
    b = 0.5
    for _ in range(1000):
        bp = (1 - h) * b + h * (1 - b)
        b = 0.2 * bp / (0.2 * bp + 0.8 * (1 - bp))
    assert beliefs[-1] == pytest.approx(b, rel=1e-3)
    assert 0 < b < h


@pytest.mark.parametrize("model", ALL_MODELS)
def test_choice_probabilities_finite_and_strictly_inside_unit_interval(model):
    env = environment_for_condition("volatile", 160, TASK)
    df = simulate_model(model, EXAMPLE_PARAMS[model], env, seed=4)
    p = df["p_choice_1"]
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    lp = reference_logps(model, EXAMPLE_PARAMS[model], df.choice, df.reward)
    assert np.isfinite(lp).all() and (lp < 0).all()


def test_perseveration_is_symmetric_under_relabelling():
    rng = np.random.default_rng(0)
    for _ in range(50):
        beta, q0, q1, rho = rng.uniform(0, 8), rng.random(), rng.random(), rng.uniform(-5, 5)
        s = rng.choice([-1.0, 1.0])
        p1 = sigmoid_np(choice_logit(beta, q0, q1, rho, s))
        p1_relabelled = sigmoid_np(choice_logit(beta, q1, q0, rho, -s))  # swap values, flip s
        assert p1_relabelled == pytest.approx(1 - p1, abs=1e-12)


def test_perseveration_model_mirrors_choices_under_relabelling():
    choices = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    rewards = np.zeros_like(choices)
    params = {"bias": 0.0, "rho": 2.0}
    lp = reference_logps("perseveration", params, choices, rewards)
    lp_mirror = reference_logps("perseveration", params, 1 - choices, rewards)
    np.testing.assert_allclose(lp, lp_mirror, atol=1e-14)
    assert lp[0] == pytest.approx(np.log(0.5))  # P(a_1 = 1) = 0.5
    assert lp[1] == pytest.approx(np.log(sigmoid_np(2.0)))  # repeat is favoured


def test_all_models_have_parameter_lists():
    assert set(MODEL_PARAMS) == set(ALL_MODELS)
