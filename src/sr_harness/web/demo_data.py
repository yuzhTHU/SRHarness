"""Deterministic sample datasets exposed by the interactive web application."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


DEMO_KINDS = frozenset({"polynomial", "grouped_parameters", "driven_ode", "kuramoto_ba"})


def _variable(description: str, axes: list[str]) -> dict[str, Any]:
    return {"description": description, "axes": axes}


def _ba_edges(rng: np.random.Generator, n_nodes: int = 10, attachments: int = 2) -> np.ndarray:
    """Build a small Barabási–Albert graph without an optional network package."""
    edges = [(left, right) for left in range(attachments + 1) for right in range(left + 1, attachments + 1)]
    degrees = np.zeros(n_nodes, dtype=float)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    for node in range(attachments + 1, n_nodes):
        probabilities = degrees[:node] / degrees[:node].sum()
        neighbors = rng.choice(node, size=attachments, replace=False, p=probabilities)
        for neighbor in neighbors:
            edges.append((int(neighbor), node))
            degrees[neighbor] += 1
            degrees[node] += 1
    return np.asarray(edges, dtype=np.int64)


def _kuramoto_sample(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    n_nodes = 10
    undirected_edges = _ba_edges(rng, n_nodes=n_nodes)
    edges = np.concatenate([undirected_edges, undirected_edges[:, ::-1]], axis=0)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    adjacency[edges[:, 0], edges[:, 1]] = 1
    adjacency[edges[:, 1], edges[:, 0]] = 1
    degrees = adjacency.sum(axis=1)
    natural_frequencies = rng.normal(0, 0.65, n_nodes)
    natural_frequencies -= natural_frequencies.mean()
    coupling = 1.35

    def derivative(_time: float, phases: np.ndarray) -> np.ndarray:
        phase_difference = phases[None, :] - phases[:, None]
        return natural_frequencies + coupling * (adjacency * np.sin(phase_difference)).sum(axis=1) / degrees

    t = np.linspace(0, 24, 481)
    initial = rng.uniform(-np.pi, np.pi, n_nodes)
    solution = solve_ivp(derivative, (float(t[0]), float(t[-1])), initial, t_eval=t, rtol=1e-9, atol=1e-11)
    if not solution.success:
        raise RuntimeError(f"Kuramoto sample integration failed: {solution.message}")
    x = solution.y.T
    dx_dt = np.stack([derivative(float(time), state) for time, state in zip(t, x)])
    omega = np.broadcast_to(natural_frequencies, x.shape).copy()
    arrays = {"t": t, "omega": omega, "x": x, "dx_dt": dx_dt, "A": edges}
    manifest = {
        "variables": {
            "omega": _variable("Angular-frequency measurement at each time and node.", ["t", "node"]),
            "x": _variable("Observed oscillator phase at each time and node.", ["t", "node"]),
            "dx_dt": _variable("Time derivative of the observed oscillator phase.", ["t", "node"]),
            "A": {
                **_variable("Directed edge representation of the observed network.", ["edge", "endpoint"]),
                "kind": "relation",
            },
        },
        "axes": {
            "t": {"file": "t.npy", "description": "Simulation time."},
            "node": {"values": [f"node{index + 1}" for index in range(n_nodes)], "description": "Network node."},
            "edge": {"size": len(edges), "description": "Network edge index."},
            "endpoint": {"values": ["target", "source"], "description": "Endpoint role in A."},
        },
        "num_nodes": n_nodes,
    }
    return arrays, manifest, "dx_dt"


def build_demo(kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """Return arrays, a context.data manifest, and the suggested target variable."""
    if kind not in DEMO_KINDS:
        raise ValueError(f"Unknown sample dataset: {kind}")
    rng = np.random.default_rng(42)
    if kind == "polynomial":
        x1 = rng.uniform(-2, 2, 240)
        x2 = rng.uniform(-1.5, 1.5, 240)
        arrays = {"x1": x1, "x2": x2, "y": 1 + x1**2 + 2*x1*x2}
        descriptions = {
            "x1": "First continuous input variable.",
            "x2": "Second continuous input variable.",
            "y": "Observed target variable.",
        }
    elif kind == "grouped_parameters":
        x1 = rng.uniform(0.5, 2.5, 300)
        x2 = rng.uniform(0.05, 3, 300)
        label = rng.choice(np.asarray(["alpha", "beta", "gamma"]), size=300)
        group_values = {"alpha": 0.45, "beta": 0.9, "gamma": 1.35}
        grouped = np.asarray([group_values[value] for value in label])
        arrays = {"x1": x1, "x2": x2, "label": label, "y": x1*np.exp(-(grouped**2)*x2)}
        descriptions = {
            "x1": "First continuous input variable.",
            "x2": "Second continuous input variable.",
            "label": "Categorical input variable with three groups.",
            "y": "Observed target variable.",
        }
    elif kind == "driven_ode":
        t = np.linspace(0, 36, 721)

        def derivative(time: float, state: np.ndarray) -> np.ndarray:
            x = state[0]
            return np.asarray([1.15*np.sin(1.35*time) + 0.42*np.sin(2.7*time) - 0.24*x - 0.075*x**3])

        solution = solve_ivp(derivative, (float(t[0]), float(t[-1])), [0.35], t_eval=t, rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError(f"ODE sample integration failed: {solution.message}")
        x = solution.y[0]
        dx_dt = np.asarray([derivative(float(time), np.asarray([value]))[0] for time, value in zip(t, x)])
        arrays = {"t": t, "x": x, "dx_dt": dx_dt}
        manifest = {
            "variables": {
                "x": _variable("Observed state trajectory.", ["t"]),
                "dx_dt": _variable("Time derivative of the observed oscillatory state x.", ["t"]),
            },
            "axes": {"t": {"file": "t.npy", "description": "Simulation time."}},
        }
        return arrays, manifest, "dx_dt"
    else:
        return _kuramoto_sample(rng)

    sample_count = len(next(iter(arrays.values())))
    manifest = {
        "variables": {
            name: _variable(description, ["sample"])
            for name, description in descriptions.items()
        },
        "axes": {"sample": {"size": sample_count, "description": "Sample index."}},
    }
    return arrays, manifest, "y"
