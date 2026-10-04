import numpy as np
import pytest

from src.predictive_checks import behavioural_summaries, reversal_window


def test_behavioural_summaries_hand_example():
    # 8 trials, reversal before array index 4 (option 0 best, then option 1)
    choices = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    rewards = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    adv = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rev = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    s = behavioural_summaries(choices, rewards, adv, rev, pre=2, post=3)
    # offsets -2,-1,0,1,2 -> indices 2,3,4,5,6 -> correct 0,1,0,1,1
    np.testing.assert_allclose(s["reversal_curve"], [0, 1, 0, 1, 1])
    assert s["accuracy"] == pytest.approx(6 / 8)  # correct = [1,1,0,1,0,1,1,1]
    # transitions: stays after reward: t1 (r0=1, stay), t4 (r3=1, stay), t6 (r5=1, stay), t7 (r6=1, stay)
    assert s["win_stay"] == pytest.approx(1.0)
    # after no reward: t2 (r1=0, switch), t3 (r2=0, switch), t5 (r4=0, switch)
    assert s["lose_switch"] == pytest.approx(1.0)
    assert s["p_repeat"] == pytest.approx(4 / 7)
    # position within stable period: [0,1,2,3,0,1,2,3] -> accuracy by position
    np.testing.assert_allclose(s["block_curve"], [(1 + 0) / 2, (1 + 1) / 2, (0 + 1) / 2, (1 + 1) / 2])


def test_reversal_window_never_crosses_next_reversal():
    assert reversal_window(40) == (10, 20)
    assert reversal_window(20) == (10, 19)
    assert reversal_window(None) == (10, 20)


def test_curve_is_nan_where_window_crosses_task_edge():
    choices = np.zeros(6, int)
    s = behavioural_summaries(choices, np.zeros(6, int), np.array([0, 1, 1, 1, 1, 1]),
                              np.array([0, 1, 0, 0, 0, 0]), pre=3, post=2)
    assert np.isnan(s["reversal_curve"][:2]).all() and s["reversal_curve"][2] == 1.0
