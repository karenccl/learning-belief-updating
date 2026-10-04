import numpy as np
import pytest

from src.config import load_config
from src.cross_validation import (
    block_masks, cross_validate, forward_chaining, sequential_predictive_log_likelihood,
)
from src.models import PRIMARY_MODELS, negative_log_likelihood, reference_logps, untransform
from src.simulate import simulate_model
from src.task import environment_for_condition

TASK = load_config()["task"]
PARAMS = {
    "fixed_rl": {"alpha": 0.3, "beta": 4.0},
    "asymmetric_rl": {"alpha_pos": 0.5, "alpha_neg": 0.15, "beta": 3.0},
    "hybrid_ph": {"eta": 0.4, "kappa": 0.7, "beta": 5.0},
    "bayes_cp": {"h": 0.1, "beta": 6.0},
}


@pytest.mark.parametrize("n,k", [(80, 2), (160, 4), (320, 8)])
def test_masks_disjoint_and_cover_every_trial(n, k):
    masks = block_masks(n, 40)
    assert len(masks) == k
    total = np.sum(masks, axis=0)
    np.testing.assert_array_equal(total, np.ones(n))  # disjoint + covering
    for m in masks:
        train = 1 - m
        assert np.all(train * m == 0) and np.all(train + m == 1)
        idx = np.flatnonzero(m)
        assert len(idx) == 40 and np.all(np.diff(idx) == 1)  # contiguous 40-trial block


@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_changing_heldout_choice_leaves_earlier_training_loss_unchanged(model):
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model(model, PARAMS[model], env, seed=1)
    c, r = df.choice.to_numpy().copy(), df.reward.to_numpy(float)
    held = block_masks(160, 40)[2]  # trials 81-120
    train = 1 - held
    t_change = 95  # a held-out trial (0-indexed)
    earlier_train = train * (np.arange(160) < t_change)
    theta = untransform(model, PARAMS[model])
    before = negative_log_likelihood(theta, c, r, earlier_train, model)
    c2 = c.copy()
    c2[t_change] = 1 - c2[t_change]
    after = negative_log_likelihood(theta, c2, r, earlier_train, model)
    assert before == after
    # and a held-out trial never contributes to the training loss directly
    lp1 = reference_logps(model, PARAMS[model], c, r)
    lp2 = reference_logps(model, PARAMS[model], c2, r)
    assert lp1[t_change] != lp2[t_change]
    np.testing.assert_array_equal(lp1[:t_change], lp2[:t_change])


@pytest.mark.parametrize("model", PRIMARY_MODELS)
def test_heldout_score_is_sum_of_heldout_trial_log_probs(model):
    env = environment_for_condition("volatile", 160, TASK)
    df = simulate_model(model, PARAMS[model], env, seed=2)
    c, r = df.choice.to_numpy(), df.reward.to_numpy(float)
    lp = reference_logps(model, PARAMS[model], c, r)
    for held in block_masks(160, 40):
        s = sequential_predictive_log_likelihood(PARAMS[model], c, r, held, model)
        assert s == pytest.approx(lp[held == 1].sum(), abs=1e-9)


def test_cross_validate_structure_and_random_benchmark_is_log_half():
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model("fixed_rl", PARAMS["fixed_rl"], env, seed=3)
    c, r = df.choice.to_numpy(), df.reward.to_numpy(float)
    rnd = cross_validate("random", c, r)
    assert rnd["heldout_ll_per_trial"] == pytest.approx(np.log(0.5))
    cv = cross_validate("fixed_rl", c, r, n_starts=5, seed_fn=lambda k: np.random.default_rng(k))
    assert len(cv["folds"]) == 4 and cv["n_scored"] == 160 and cv["n_failed_folds"] == 0
    assert cv["heldout_ll_per_trial"] > np.log(0.5)  # learner beats random on its own data
    fc = forward_chaining("fixed_rl", c, r, n_starts=5, seed_fn=lambda k: np.random.default_rng(k))
    assert len(fc["folds"]) == 3 and fc["n_scored"] == 120
