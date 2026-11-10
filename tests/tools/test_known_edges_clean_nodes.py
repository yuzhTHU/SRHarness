"""Clean-node coefficient lock must not change an exact all-node fit."""
import types
import unittest
import numpy as np

from sr_harness.tools._hypergraph_known_edge_common import (
    column,
    consensus_shared_coefficients,
    fit,
    normalize_terms,
)
from sr_harness.tools.hypergraph_formula_refine import HypergraphFormulaRefineTool


def _context():
    rng = np.random.default_rng(7)
    t = np.linspace(0, 0.4, 40)
    n_traj, n_node = 3, 4
    x = rng.uniform(-1.2, 1.2, (n_traj, t.size, n_node))
    pairwise = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [0, 2], [2, 0]])
    ctx = types.SimpleNamespace(num_nodes=n_node, relation_names=['pairwise', 'triplet'])
    ctx.data = {
        'x': x, 'dx_dt': np.zeros_like(x),
        'pairwise': pairwise, 'triplet': np.zeros((0, 3), dtype=int),
    }
    ctx.args = types.SimpleNamespace(known_edges={
        'config': dict(pairwise='pairwise', triplet='triplet', num_nodes=n_node),
        'observations': dict(x=x, t=np.broadcast_to(t, (n_traj, t.size))),
        'weights': {},
        'runtime': {'candidates': {}, 'gates': {}},
    })
    ctx.data['dx_dt'] = column(ctx, x, {'channel': 'G', 'kernel': 'sin(xj-xi)'})
    return ctx


class CleanNodeLock(unittest.TestCase):
    def test_true_graph_does_not_lock_away_from_global_fit(self):
        ctx = _context()
        terms = normalize_terms(ctx, [{'channel': 'G', 'kernel': 'sin(xj-xi)'}])
        coef, metrics = fit(ctx, terms, 'none')
        self.assertLess(metrics['derivative_rmse'], 1e-12)
        self.assertIsNone(consensus_shared_coefficients(ctx, terms, coef))

    def test_dirty_nodes_do_not_pull_amplitude(self):
        ctx = _context()
        ctx.data['dx_dt'][..., 3] += 0.8 * np.sin(ctx.data['x'][..., 3])
        terms = normalize_terms(ctx, [{'channel': 'G', 'kernel': 'sin(xj-xi)'}])
        coef, _ = fit(ctx, terms, 'none')
        locked = consensus_shared_coefficients(ctx, terms, coef)
        self.assertIsNotNone(locked)
        self.assertAlmostEqual(float(locked[0]), 1.0, places=5)
        tool = HypergraphFormulaRefineTool()
        tool.context = ctx
        from sr_harness.tools._hypergraph_known_edge_common import save
        cid = save(ctx, terms, coef, 'none', 'screen', {})
        result = tool.execute(cid)
        self.assertTrue(any(item.get('action') == 'lock_clean_nodes' for item in result.get('trace') or []))


if __name__ == '__main__':
    unittest.main()
