import numpy as np
import pytest

from src.config import load_config
from src.models import make_agent
from src.simulate import simulate_model
from src.task import environment_for_condition, make_environment, reversal_schedule

CFG = load_config()
TASK = CFG["task"]


@pytest.mark.parametrize(
    "condition,n,expected",
    [
        ("stable", 80, [60]), ("stable", 160, [120]), ("stable", 320, [240]),
        ("moderate", 80, [40]), ("moderate", 160, [40, 80, 120]),
        ("moderate", 320, [40, 80, 120, 160, 200, 240, 280]),
        ("volatile", 80, [20, 40, 60]), ("volatile", 160, [20, 40, 60, 80, 100, 120, 140]),
    ],
)
def test_reversal_schedules_match_spec(condition, n, expected):
    assert reversal_schedule(condition, n, TASK) == expected


def test_volatile_320_has_reversal_every_20():
    assert reversal_schedule("volatile", 320, TASK) == list(range(20, 320, 20))


def test_probabilities_swap_exactly_on_scheduled_reversals():
    env = make_environment(160, 0.8, 0.2, [40, 80, 120], 0)
    p0, p1 = env["p_option_0"], env["p_option_1"]
    for k in [40, 80, 120]:  # "after trial k": array index k-1 vs k
        assert p0[k] == p1[k - 1] and p1[k] == p0[k - 1]
        assert env["reversal"][k] == 1
    # Initial state: option 0 advantageous
    assert p0[0] == 0.8 and p1[0] == 0.2 and env["advantageous_option"][0] == 0


def test_probabilities_unchanged_on_non_reversal_trials():
    env = make_environment(160, 0.8, 0.2, [40, 80, 120], 0)
    changed = np.flatnonzero(np.diff(env["p_option_0"]) != 0) + 1
    assert changed.tolist() == [40, 80, 120]
    assert env["reversal"].sum() == 3
    np.testing.assert_allclose(env["p_option_0"] + env["p_option_1"], 1.0)


def test_advantageous_option_consistent_with_probabilities():
    env = environment_for_condition("volatile", 160, TASK)
    adv = np.where(env["p_option_0"] > env["p_option_1"], 0, 1)
    np.testing.assert_array_equal(adv, env["advantageous_option"])


def test_rewards_binary_and_sampled_from_chosen_option():
    env = make_environment(4000, 0.8, 0.2, [2000], 0)
    df = simulate_model("random", {}, env, seed=3)
    assert set(df["reward"].unique()) <= {0, 1}
    assert set(df["choice"].unique()) <= {0, 1}
    p_chosen = np.where(df["choice"] == 1, df["p_option_1"], df["p_option_0"])
    for p in (0.8, 0.2):
        sel = p_chosen == p
        rate = df.loc[sel, "reward"].mean()
        se = np.sqrt(p * (1 - p) / sel.sum())
        assert abs(rate - p) < 4 * se


def test_simulation_is_reproducible_from_seed():
    env = environment_for_condition("moderate", 160, TASK)
    params = {"alpha": 0.3, "beta": 4.0}
    a = simulate_model("fixed_rl", params, env, seed=np.random.SeedSequence(7))
    b = simulate_model("fixed_rl", params, env, seed=np.random.SeedSequence(7))
    assert a.equals(b)


def test_trial_table_has_required_columns():
    env = environment_for_condition("moderate", 160, TASK)
    df = simulate_model("hybrid_ph", {"eta": 0.3, "kappa": 0.5, "beta": 3.0}, env, seed=1)
    required = {"participant_id", "trial", "condition", "block", "choice", "reward", "p_option_0",
                "p_option_1", "advantageous_option", "reversal", "true_model", "true_parameters",
                "cv_block"}
    assert required <= set(df.columns)
    assert df["cv_block"].tolist() == list(np.repeat(np.arange(4), 40))
    assert {"Q0", "Q1", "prediction_error", "alpha_t", "associability"} <= set(df.columns)


def test_unchosen_value_is_not_updated_in_simulation():
    agent = make_agent("fixed_rl", {"alpha": 0.3, "beta": 4.0})
    agent.logit(0, 0.0)
    agent.update(0, 1)
    assert agent.Q[1] == 0.5


def test_agent_seed_equals_spawned_child_and_is_unique():
    from src.seeds import agent_index, agent_seed

    idx = agent_index("smoke", "fixed_rl", "moderate", 50, 3)
    child = np.random.SeedSequence(123).spawn(idx + 1)[idx]
    ours = agent_seed(123, "smoke", "fixed_rl", "moderate", 50, 3)
    np.testing.assert_array_equal(child.generate_state(4), ours.generate_state(4))
    keys = {agent_index(s, m, c, n, a)
            for s in ("parameter_recovery", "model_recovery")
            for m in ("fixed_rl", "bayes_cp") for c in ("stable", "volatile")
            for n in (80, 160) for a in (0, 49)}
    assert len(keys) == 2 * 2 * 2 * 2 * 2


def test_simulate_arrays_reproduces_simulate_model():
    from src.simulate import simulate_arrays

    env = environment_for_condition("volatile", 160, TASK)
    for model, params in [("fixed_rl", {"alpha": 0.3, "beta": 4.0}), ("bayes_cp", {"h": 0.1, "beta": 5.0}),
                          ("hybrid_ph", {"eta": 0.3, "kappa": 0.6, "beta": 3.0})]:
        df = simulate_model(model, params, env, seed=np.random.SeedSequence(3))
        c, r = simulate_arrays(model, params, env, seed=np.random.SeedSequence(3))
        np.testing.assert_array_equal(df.choice.to_numpy(), c)
        np.testing.assert_array_equal(df.reward.to_numpy(), r)
