"""Construct small in-memory arrays and run multivariate known-edges procedures."""
import types
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from sr_harness.tools import _hypergraph_multivariate as v


def context(names=('u', 'v'), trajectories=3):
    rng = np.random.default_rng(19)
    c = types.SimpleNamespace(num_nodes=3, relation_names=['pairwise', 'triplet'])
    c.data = {name: rng.uniform(-1.8, 1.8, (trajectories, 60, 3)) for name in names}
    c.data.update(pairwise=np.array([[0, 1], [1, 2]]), triplet=np.array([[2, 0, 1]]))
    for name in names:
        c.data['d' + name + '_dt'] = np.zeros_like(c.data[name])
    c.args = types.SimpleNamespace(known_edges={
        'config': dict(
            state_dim=len(names), components=list(names), num_nodes=3,
            pairwise='pairwise', triplet='triplet',
        ),
        'observations': {},
        'weights': {},
    })
    return c


def seed(c, terms=None):
    per = {}
    for name in v.component_names(c):
        ts = (terms or {}).get(name, [])
        co, _ = v.vector_fit(c, ts, name)
        per[name] = dict(terms=ts, coefficients=co.tolist(), node_mode='none')
    return v.multivariate_save(c, per, 'screen')


class VectorProcedures(unittest.TestCase):
    def test_local_cross(self):
        for bilinear in (False, True):
            c = context()
            c.data['du_dt'] = c.data['v'] if not bilinear else c.data['u'] * c.data['v']
            r = v.refine_multivariate(c, seed(c))
            expected = 'local_bilinear_cross' if bilinear else 'local_linear_cross'
            self.assertTrue(any(t.get('family') == expected and t['action'] == 'vector_family_contrast' for t in r['trace']))
            self.assertEqual(r['components']['v']['zero_slots'], ['G', 'H'])
            self.assertFalse(r['can_submit'])

    def test_single_channel_families(self):
        for family, records in v.MULTIVARIATE_COUPLING_FAMILIES:
            c = context()
            # drive v from u, leaving the other output identically zero
            c.data['dv_dt'] = sum(v.vector_column(c, t) for t in records)
            r = v.refine_multivariate(c, seed(c))
            self.assertTrue(any(t.get('family') == family+':u' and t['action']=='vector_family_contrast' for t in r['trace']))
            self.assertEqual(r['components']['u']['zero_slots'], ['G', 'H'])
            self.assertLess(r['components']['v']['derivative_rmse'], 1e-10)

    def test_host_component_names(self):
        c = context(('a', 'b', 'c'))
        c.data['da_dt'] = c.data['c']
        r = v.refine_multivariate(c, seed(c))
        self.assertLess(r['components']['a']['derivative_rmse'], 1e-10)

    def test_node_guard(self):
        c = context()
        offsets = np.random.default_rng(13).normal(size=3)
        c.data['du_dt'] = np.broadcast_to(offsets, c.data['u'].shape).copy()
        cid = seed(c, {'u': [{'channel':'F', 'kernel':'1'}]})
        d = v.diagnose_multivariate(c, cid)
        self.assertEqual(d['per_variable']['u']['recommended_node_mode'], 'constant')
        self.assertEqual(d['per_component']['u']['recommended_node_mode'], 'constant')
        r = v.node_fit_multivariate(c, cid, 'constant', 'Repeated offsets across independent synthetic trajectories')
        self.assertEqual(r['status'], 'accepted_for_refinement')
        self.assertEqual(sum(b['node_mode']!='none' for b in r['components'].values()), 1)
        with self.assertRaises(ValueError): v.node_fit_multivariate(c, r['candidate_id'], 'linear', 'Repeated synthetic evidence')
        self.assertTrue(v.refine_multivariate(c, r['candidate_id'])['short_review'])
        one = context(trajectories=1)
        one.data['du_dt'] = np.broadcast_to(offsets, one.data['u'].shape).copy()
        self.assertEqual(v.node_fit_multivariate(one, seed(one), 'constant', 'Synthetic single trajectory evidence')['status'], 'rejected')

    def test_replacement_rollback(self):
        c = context(); c.data['du_dt'] = .1234567*c.data['v']
        cid = seed(c)
        r = v.refine_multivariate(c, cid, [{'component':'u', 'channel':'F', 'kernel':'vi'}],
            'Local cross residual on the output component', simplify_coefficients=False,
            coefficient_cutoff=0, contribution_cutoff=0)
        self.assertIn('rollback_id', r)
        self.assertLess(r['components']['u']['derivative_rmse'], 1e-12)
        self.assertFalse(r['can_submit'])

    def test_no_holdout_no_family_switch(self):
        c = context(trajectories=1); c.data['du_dt']=c.data['v']
        r=v.refine_multivariate(c, seed(c))
        self.assertFalse(any(t['action']=='vector_family_contrast' for t in r['trace']))

    def test_full_grid_gate_and_stale_id(self):
        c=context(); t=np.linspace(0, .4, 25)
        rng=np.random.default_rng(8)
        xs=[]
        for _ in range(3):
            initial=rng.normal(size=(3,2))
            sol=solve_ivp(lambda _, z: (-.3*z), (t[0],t[-1]), initial.ravel(), t_eval=t, rtol=1e-11, atol=1e-13)
            xs.append(sol.y.T.reshape(-1,3,2))
        x=np.asarray(xs)
        for a,name in enumerate(('u','v')):
            c.data[name]=x[...,a]; c.data['d'+name+'_dt']=-.3*x[...,a]
        c.args.known_edges['observations']=dict(x=x,t=t)
        r=v.refine_multivariate(c, seed(c, {n:[dict(channel='F',kernel=n+'i')] for n in ('u','v')}))
        gate=v.reintegrate_multivariate(c,r['candidate_id'])
        self.assertTrue(gate['can_submit']); self.assertEqual(set(gate['per_variable_rmse']),{'u','v'})
        self.assertEqual(set(gate['per_component_rmse']),{'u','v'})
        c.data['du_dt']=c.data['du_dt']+.1
        with self.assertRaises(ValueError): v.reintegrate_multivariate(c,r['candidate_id'])

    def test_default_coupling_has_no_high_order(self):
        self.assertEqual(len(v._coupling_library()), 6)
        self.assertFalse(any('**' in t['kernel'] for t in v._coupling_library()))
        self.assertTrue(any(name == 'cubic_ordered' for name, _ in v.MULTIVARIATE_POLY3_FAMILIES))

    def test_cubic_ordered_single_channel(self):
        c = context()
        recs = [
            {'channel': 'G', 'kernel': 'uj - ui'},
            {'channel': 'H', 'kernel': 'uj * uk**2'},
            {'channel': 'H', 'kernel': 'ui**3'},
        ]
        mapped = [{'channel': t['channel'], 'kernel': t['kernel']} for t in recs]
        c.data['du_dt'] = sum(v.vector_column(c, t) for t in mapped)
        r = v.refine_multivariate(c, seed(c))
        self.assertTrue(any(
            t.get('action') == 'vector_family_contrast' and str(t.get('family', '')).startswith('cubic_')
            for t in r['trace']
        ))
        self.assertLess(r['components']['u']['derivative_rmse'], 1e-8)
        self.assertEqual(r['components']['v']['zero_slots'], ['G', 'H'])

    def test_cubic_ordered_three_components_budget(self):
        c = context(('u', 'v', 'w'))
        recs = [
            {'channel': 'G', 'kernel': 'uj - ui'},
            {'channel': 'H', 'kernel': 'uj * uk**2'},
            {'channel': 'H', 'kernel': 'ui**3'},
        ]
        c.data['du_dt'] = sum(v.vector_column(c, t) for t in recs)
        r = v.refine_multivariate(c, seed(c))
        self.assertTrue(any(
            t.get('action') == 'vector_family_contrast' and t.get('family') == 'cubic_ordered:u'
            for t in r['trace']
        ))
        self.assertLess(r['components']['u']['derivative_rmse'], 1e-8)

    def test_agent_display_does_not_require_train_split(self):
        from sr_harness.tools.evaluate_formula import SubmitFormulaTool
        from sr_harness.tools.hypergraph_formula_screen import HypergraphFormulaScreenTool
        from sr_harness.tools._hypergraph_known_edge_common import require_submission

        screen = HypergraphFormulaScreenTool.format_result_dict({
            'known_edges': {'candidate_id': 'abc', 'multivariate': True, 'vector': True, 'can_submit': False},
            'screen_configuration': {'multivariate': True, 'vector': True, 'components': ['u'], 'max_terms': 6},
            'multivariate': True,
            'vector': True,
            'components': {'u': {'F': '(1)*(vi)', 'G': '0', 'H': '0', 'derivative_rmse': 0.1}},
            'formula': 'du_dt = (1) * (v_i) + sum G + sum H',
            'candidate_id': 'abc',
            'can_submit': False,
        })
        self.assertIn('candidate_id', screen)
        submitted = SubmitFormulaTool.format_result_dict({
            'formula': 'du_dt = (1) * (v_i)',
            'F': '((1) * (v_i), 0)',
            'G': '(0, 0)',
            'H': '(0, 0)',
            'resim_rmse': 1e-9,
            'is_candidate': True,
            'data_split_results': {'validation': {'metrics': {'mse': 1e-9}}},
        })
        self.assertIn('resim_rmse', submitted)
        ctx = context()
        ctx.args.known_edges['runtime'] = {'candidates': {}, 'gates': {}}
        with self.assertRaises(ValueError):
            require_submission(ctx, 'v')

    def test_illegal_residual_allowance_is_clamped(self):
        c = context()
        c.data['du_dt'] = c.data['v']
        result = v.refine_multivariate(c, seed(c), residual_allowance=0.05)
        self.assertIn('candidate_id', result)
        self.assertLess(result['components']['u']['derivative_rmse'], 1e-10)

    def test_rounds_near_one_cubic_coefficients(self):
        c = context()
        recs = [
            {'channel': 'H', 'kernel': 'uj * uk**2'},
            {'channel': 'H', 'kernel': 'ui**3'},
        ]
        amp = 0.999988544904
        c.data['du_dt'] = amp * v.vector_column(c, recs[0]) + (-amp) * v.vector_column(c, recs[1])
        per = {
            'u': dict(terms=recs, coefficients=[amp, -amp], node_mode='none'),
            'v': dict(terms=[], coefficients=[], node_mode='none'),
        }
        result = v.refine_multivariate(c, v.multivariate_save(c, per, 'screen'))
        coef = np.asarray(v.variable_blocks(v.get(c, result['candidate_id']))['u']['coefficients'], float)
        self.assertTrue(any(item.get('action') == 'round' for item in result['trace']))
        self.assertTrue(np.allclose(np.abs(coef[:2]), 1.0, atol=1e-12))

    def test_node_fit_reintegrate_can_submit(self):
        from sr_harness.tools._hypergraph_known_edge_common import require_submission
        c = context()
        t = np.linspace(0, 0.4, 25)
        rng = np.random.default_rng(8)
        offsets = rng.normal(size=3)
        xs = []
        for _ in range(3):
            initial = rng.normal(size=(3, 2))
            def fun(_, z, drive=offsets):
                y = z.reshape(3, 2)
                return np.column_stack([drive, -0.3 * y[:, 1]]).ravel()
            sol = solve_ivp(fun, (t[0], t[-1]), initial.ravel(), t_eval=t, rtol=1e-11, atol=1e-13)
            xs.append(sol.y.T.reshape(-1, 3, 2))
        x = np.asarray(xs)
        c.data['u'], c.data['v'] = x[..., 0], x[..., 1]
        c.data['du_dt'] = np.broadcast_to(offsets, c.data['u'].shape).copy()
        c.data['dv_dt'] = -0.3 * c.data['v']
        c.args.known_edges['observations'] = dict(x=x, t=t)
        cid = seed(c, {
            'u': [{'channel': 'F', 'kernel': '1'}],
            'v': [{'channel': 'F', 'kernel': 'vi'}],
        })
        fitted = v.node_fit_multivariate(
            c, cid, 'constant', 'Repeated offsets across independent synthetic trajectories',
        )
        self.assertEqual(fitted['status'], 'accepted_for_refinement')
        gate = v.reintegrate_multivariate(c, fitted['candidate_id'])
        self.assertEqual(gate['status'], 'passed')
        self.assertTrue(gate['can_submit'])
        self.assertIn('submit_formula', gate['next_action'])
        require_submission(c, fitted['candidate_id'])
        with self.assertRaises(ValueError) as err:
            require_submission(c, 'v')
        self.assertIn(fitted['candidate_id'], str(err.exception))

if __name__ == '__main__': unittest.main()
