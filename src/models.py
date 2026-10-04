"""Model space: parameter transforms, NumPy reference agents, JAX likelihoods.

Two implementations of every model live here:

* ``ReferenceAgent`` subclasses: plain Python/NumPy, one trial at a time. They
  are used for simulation *and* for the reference likelihood, so simulation and
  fitting share exactly the same update equations.
* ``_core_logps_*``: JAX versions written with ``jax.lax.scan``. They take
  natural (transformed) parameters and return per-trial log probabilities of
  the observed choices. Unit tests check they match the references to 1e-8.

All choices are coded 0/1. The shared choice rule is
    P(a_t = 1) = logistic(beta * (Q1 - Q0) + rho * s_{t-1}),
with rho = 0 in the four primary models.
"""
from __future__ import annotations

import os

# Tiny scans gain nothing from intra-op threading; keep each process single-threaded
# so that joblib can parallelise across agents instead.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

# The Bayesian learner is given the true generative reward probabilities.
P_HIGH = 0.8
P_LOW = 0.2
BETA_FLOOR = 1e-4

MODEL_PARAMS: dict[str, tuple[str, ...]] = {
    "random": (),
    "perseveration": ("bias", "rho"),
    "fixed_rl": ("alpha", "beta"),
    "asymmetric_rl": ("alpha_pos", "alpha_neg", "beta"),
    "hybrid_ph": ("eta", "kappa", "beta"),
    "bayes_cp": ("h", "beta"),
}
PRIMARY_MODELS = ("fixed_rl", "asymmetric_rl", "hybrid_ph", "bayes_cp")
BENCHMARK_MODELS = ("random", "perseveration")
ALL_MODELS = PRIMARY_MODELS + BENCHMARK_MODELS

MODEL_LABELS = {
    "random": "Random",
    "perseveration": "Perseveration",
    "fixed_rl": "Fixed RL",
    "asymmetric_rl": "Asymmetric RL",
    "hybrid_ph": "PH hybrid",
    "bayes_cp": "Bayesian CP",
}

PARAM_TRANSFORM = {
    "alpha": "logistic",
    "alpha_pos": "logistic",
    "alpha_neg": "logistic",
    "eta": "logistic",
    "kappa": "logistic",
    "h": "logistic",
    "beta": "softplus",
    "rho": "tanh5",
    "bias": "identity",
}

# Post-hoc sensitivity variant (not the locked design): beta bounded above at BETA_MAX
# and the Bayesian hazard restricted to (0, 0.5), i.e. no "expect alternation" regime.
BETA_MAX = 20.0
TRANSFORM_VARIANTS = {
    "locked": PARAM_TRANSFORM,
    "bounded": {**PARAM_TRANSFORM, "beta": "bounded_beta", "h": "half_logistic"},
}


# ---------------------------------------------------------------------------
# Parameter transforms
# ---------------------------------------------------------------------------
def _forward(kind: str, x):
    if kind == "logistic":
        return jax.nn.sigmoid(x)
    if kind == "softplus":
        return jax.nn.softplus(x) + BETA_FLOOR
    if kind == "tanh5":
        return 5.0 * jnp.tanh(x)
    if kind == "identity":
        return x
    if kind == "bounded_beta":
        return BETA_FLOOR + (BETA_MAX - BETA_FLOOR) * jax.nn.sigmoid(x)
    if kind == "half_logistic":
        return 0.5 * jax.nn.sigmoid(x)
    raise ValueError(kind)


def _inverse(kind: str, y: float) -> float:
    if kind == "logistic":
        return float(np.log(y) - np.log1p(-y))
    if kind == "softplus":
        z = y - BETA_FLOOR
        return float(z + np.log(-np.expm1(-z)))  # stable log(expm1(z))
    if kind == "tanh5":
        return float(np.arctanh(y / 5.0))
    if kind == "identity":
        return float(y)
    if kind == "bounded_beta":
        return _inverse("logistic", (y - BETA_FLOOR) / (BETA_MAX - BETA_FLOOR))
    if kind == "half_logistic":
        return _inverse("logistic", 2.0 * y)
    raise ValueError(kind)


def transform(model_name: str, theta, variant: str = "locked"):
    """Map unconstrained ``theta`` to natural parameters (JAX array, model order)."""
    names = MODEL_PARAMS[model_name]
    if not names:
        return jnp.zeros(0)
    kinds = TRANSFORM_VARIANTS[variant]
    return jnp.stack([_forward(kinds[n], theta[i]) for i, n in enumerate(names)])


def untransform(model_name: str, params: dict, variant: str = "locked") -> np.ndarray:
    """Map a dict of natural parameters to unconstrained ``theta``."""
    kinds = TRANSFORM_VARIANTS[variant]
    return np.array([_inverse(kinds[n], params[n]) for n in MODEL_PARAMS[model_name]], dtype=float)


def params_to_dict(model_name: str, values) -> dict:
    return {n: float(v) for n, v in zip(MODEL_PARAMS[model_name], np.asarray(values))}


def dict_to_array(model_name: str, params: dict) -> np.ndarray:
    return np.array([params[n] for n in MODEL_PARAMS[model_name]], dtype=float)


# ---------------------------------------------------------------------------
# Shared choice rule (NumPy)
# ---------------------------------------------------------------------------
def choice_logit(beta: float, q0: float, q1: float, rho: float = 0.0, prev_s: float = 0.0) -> float:
    """Logit of P(a=1) under the shared choice rule, with symmetric perseveration."""
    return beta * (q1 - q0) + rho * prev_s


def log_sigmoid_np(x: float) -> float:
    return -float(np.logaddexp(0.0, -x))


def sigmoid_np(x: float) -> float:
    return float(np.exp(log_sigmoid_np(x)))


# ---------------------------------------------------------------------------
# NumPy reference agents
# ---------------------------------------------------------------------------
class ReferenceAgent:
    """One simulated/fitted learner. Call ``logit`` once per trial, then ``update``."""

    def __init__(self, params: dict):
        self.params = params

    def logit(self, t: int, prev_s: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def latents(self) -> dict:
        return {}

    def update(self, a: int, r: int) -> dict:
        return {}


class RandomAgent(ReferenceAgent):
    def logit(self, t, prev_s):
        return 0.0


class PerseverationAgent(ReferenceAgent):
    def logit(self, t, prev_s):
        if t == 0:
            return 0.0  # P(a_1 = 1) = 0.5 on the first trial
        return self.params["bias"] + self.params["rho"] * prev_s


class FixedRLAgent(ReferenceAgent):
    def __init__(self, params):
        super().__init__(params)
        self.Q = np.array([0.5, 0.5])

    def logit(self, t, prev_s):
        return choice_logit(self.params["beta"], self.Q[0], self.Q[1])

    def latents(self):
        return {"Q0": self.Q[0], "Q1": self.Q[1]}

    def learning_rate(self, delta: float) -> float:
        return self.params["alpha"]

    def update(self, a, r):
        delta = r - self.Q[a]
        alpha_t = self.learning_rate(delta)
        self.Q[a] = self.Q[a] + alpha_t * delta
        return {"prediction_error": delta, "alpha_t": alpha_t}


class AsymmetricRLAgent(FixedRLAgent):
    def learning_rate(self, delta):
        return self.params["alpha_pos"] if delta >= 0 else self.params["alpha_neg"]


class HybridPHAgent(FixedRLAgent):
    """Pearce-Hall-style hybrid: alpha_t = kappa * A_t, A updated after the value update."""

    def __init__(self, params):
        super().__init__(params)
        self.A = 0.5

    def latents(self):
        return {"Q0": self.Q[0], "Q1": self.Q[1], "associability": self.A}

    def update(self, a, r):
        delta = r - self.Q[a]
        alpha_t = self.params["kappa"] * self.A  # uses associability *before* this outcome
        self.Q[a] = self.Q[a] + alpha_t * delta
        self.A = self.A + self.params["eta"] * (abs(delta) - self.A)
        return {"prediction_error": delta, "alpha_t": alpha_t}


class BayesCPAgent(ReferenceAgent):
    """Bayesian change-point learner over the identity of the advantageous option.

    ``b_post`` is P(z = 0 | data up to t); ``b_pred`` is the predictive belief
    after the hazard transition, used for the choice on trial t.
    """

    def __init__(self, params, p_high=P_HIGH, p_low=P_LOW):
        super().__init__(params)
        self.p_high, self.p_low = p_high, p_low
        self.b_post = 0.5
        self.b_pred = None

    def logit(self, t, prev_s):
        h = self.params["h"]
        self.b_pred = (1 - h) * self.b_post + h * (1 - self.b_post)
        q0, q1 = self._q(self.b_pred)
        return choice_logit(self.params["beta"], q0, q1)

    def _q(self, b):
        return b * self.p_high + (1 - b) * self.p_low, (1 - b) * self.p_high + b * self.p_low

    def latents(self):
        q0, q1 = self._q(self.b_pred)
        return {"Q0": q0, "Q1": q1, "belief_option_0": self.b_pred}

    def likelihoods(self, a, r):
        """Return (l0, l1) = P(r | a, z=0), P(r | a, z=1)."""
        p_if_z0 = self.p_high if a == 0 else self.p_low
        p_if_z1 = self.p_high if a == 1 else self.p_low
        if r == 1:
            return p_if_z0, p_if_z1
        return 1 - p_if_z0, 1 - p_if_z1

    def update(self, a, r):
        b = self.b_pred
        q = self._q(b)[a]
        l0, l1 = self.likelihoods(a, r)
        self.b_post = l0 * b / (l0 * b + l1 * (1 - b))
        return {"prediction_error": r - q}


AGENTS = {
    "random": RandomAgent,
    "perseveration": PerseverationAgent,
    "fixed_rl": FixedRLAgent,
    "asymmetric_rl": AsymmetricRLAgent,
    "hybrid_ph": HybridPHAgent,
    "bayes_cp": BayesCPAgent,
}


def make_agent(model_name: str, params: dict) -> ReferenceAgent:
    return AGENTS[model_name](params)


def reference_logps(model_name: str, params: dict, choices, rewards) -> np.ndarray:
    """Plain-Python per-trial log P(observed choice). The reference for JAX tests."""
    agent = make_agent(model_name, params)
    out = np.empty(len(choices))
    prev_s = 0.0
    for t, (a, r) in enumerate(zip(np.asarray(choices, int), np.asarray(rewards, int))):
        x = agent.logit(t, prev_s)
        out[t] = log_sigmoid_np(x) if a == 1 else log_sigmoid_np(-x)
        agent.update(int(a), int(r))
        prev_s = 2.0 * a - 1.0
    return out


def reference_nll(model_name: str, params: dict, choices, rewards, score_mask=None) -> float:
    lp = reference_logps(model_name, params, choices, rewards)
    mask = np.ones_like(lp) if score_mask is None else np.asarray(score_mask, float)
    return float(-np.sum(mask * lp))


# ---------------------------------------------------------------------------
# JAX likelihoods (lax.scan over trials)
# ---------------------------------------------------------------------------
def _logp_choice(logit, a):
    return jnp.where(a == 1, jax.nn.log_sigmoid(logit), jax.nn.log_sigmoid(-logit))


def _core_random(p, choices, rewards):
    return jnp.full(choices.shape, jnp.log(0.5))


def _core_perseveration(p, choices, rewards):
    bias, rho = p[0], p[1]
    t_idx = jnp.arange(choices.shape[0])

    def step(prev_s, x):
        a, t = x
        logit = jnp.where(t == 0, 0.0, bias + rho * prev_s)
        return 2.0 * a - 1.0, _logp_choice(logit, a)

    _, lp = jax.lax.scan(step, jnp.asarray(0.0), (choices, t_idx))
    return lp


def _core_fixed_rl(p, choices, rewards):
    alpha, beta = p[0], p[1]

    def step(Q, x):
        a, r = x
        logp = _logp_choice(beta * (Q[1] - Q[0]), a)
        delta = r - Q[a]
        return Q.at[a].add(alpha * delta), logp

    _, lp = jax.lax.scan(step, jnp.array([0.5, 0.5]), (choices, rewards))
    return lp


def _core_asymmetric_rl(p, choices, rewards):
    alpha_pos, alpha_neg, beta = p[0], p[1], p[2]

    def step(Q, x):
        a, r = x
        logp = _logp_choice(beta * (Q[1] - Q[0]), a)
        delta = r - Q[a]
        alpha_t = jnp.where(delta >= 0, alpha_pos, alpha_neg)
        return Q.at[a].add(alpha_t * delta), logp

    _, lp = jax.lax.scan(step, jnp.array([0.5, 0.5]), (choices, rewards))
    return lp


def _core_hybrid_ph(p, choices, rewards):
    eta, kappa, beta = p[0], p[1], p[2]

    def step(carry, x):
        Q, A = carry
        a, r = x
        logp = _logp_choice(beta * (Q[1] - Q[0]), a)
        delta = r - Q[a]
        Q = Q.at[a].add(kappa * A * delta)  # learning rate uses A before this outcome
        A = A + eta * (jnp.abs(delta) - A)
        return (Q, A), logp

    _, lp = jax.lax.scan(step, (jnp.array([0.5, 0.5]), jnp.asarray(0.5)), (choices, rewards))
    return lp


def _core_bayes_cp(p, choices, rewards):
    h, beta = p[0], p[1]

    def step(b_post, x):
        a, r = x
        b = (1 - h) * b_post + h * (1 - b_post)  # transition, then predict
        q0 = b * P_HIGH + (1 - b) * P_LOW
        q1 = (1 - b) * P_HIGH + b * P_LOW
        logp = _logp_choice(beta * (q1 - q0), a)
        p_if_z0 = jnp.where(a == 0, P_HIGH, P_LOW)
        p_if_z1 = jnp.where(a == 1, P_HIGH, P_LOW)
        l0 = jnp.where(r == 1, p_if_z0, 1 - p_if_z0)
        l1 = jnp.where(r == 1, p_if_z1, 1 - p_if_z1)
        b_new = l0 * b / (l0 * b + l1 * (1 - b))  # then update
        return b_new, logp

    _, lp = jax.lax.scan(step, jnp.asarray(0.5), (choices, rewards))
    return lp


_CORES = {
    "random": _core_random,
    "perseveration": _core_perseveration,
    "fixed_rl": _core_fixed_rl,
    "asymmetric_rl": _core_asymmetric_rl,
    "hybrid_ph": _core_hybrid_ph,
    "bayes_cp": _core_bayes_cp,
}


def logps_from_params(model_name: str, params, choices, rewards):
    """Per-trial log P(observed choice) at natural parameters (JAX)."""
    return _CORES[model_name](jnp.asarray(params, dtype=jnp.float64), choices, rewards)


def _nll(theta, choices, rewards, score_mask, model_name, variant="locked"):
    p = transform(model_name, theta, variant)
    return -jnp.sum(score_mask * _CORES[model_name](p, choices, rewards))


def negative_log_likelihood(theta, choices, rewards, score_mask, model_name: str,
                            variant: str = "locked") -> float:
    """Negative sequential log likelihood, summed only over trials where score_mask == 1."""
    return float(_jitted_nll(model_name, variant)(jnp.asarray(theta, jnp.float64),
                                                  *_as_jax(choices, rewards, score_mask)))


def _as_jax(choices, rewards, mask):
    return (
        jnp.asarray(choices, dtype=jnp.int32),
        jnp.asarray(rewards, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
    )


_NLL_CACHE: dict = {}
_VG_CACHE: dict = {}
_LOGP_CACHE: dict = {}


def _jitted_nll(model_name, variant="locked"):
    key = (model_name, variant)
    if key not in _NLL_CACHE:
        _NLL_CACHE[key] = jax.jit(partial(_nll, model_name=model_name, variant=variant))
    return _NLL_CACHE[key]


def nll_and_grad_fn(model_name: str, variant: str = "locked"):
    """JIT-compiled ``(theta, choices, rewards, mask) -> (nll, grad)``."""
    key = (model_name, variant)
    if key not in _VG_CACHE:
        _VG_CACHE[key] = jax.jit(jax.value_and_grad(partial(_nll, model_name=model_name, variant=variant)))
    return _VG_CACHE[key]


def logps_fn(model_name: str):
    """JIT-compiled ``(params, choices, rewards) -> per-trial log p``."""
    if model_name not in _LOGP_CACHE:
        _LOGP_CACHE[model_name] = jax.jit(_CORES[model_name])
    return _LOGP_CACHE[model_name]
