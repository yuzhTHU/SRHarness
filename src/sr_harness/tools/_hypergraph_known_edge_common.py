"""Known-edges runtime helpers. No file/network IO, no support mutations.

NEW host contract (not an existing engine API): context.data['_known_edges_config']
may supply pairwise/triplet names and triplet_counting='rows'|'ordered_symmetric'.
Row-aligned weights live in data[relation + '_weights']; missing means ones.
Full, uncropped observations live in data['_known_edges_observations']={'x','t'}.
The host must provide target-first expanded relations; we never install/expand edges.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import operator
import re
import numpy as np

FUN = {k: getattr(np, k) for k in ('sin', 'cos', 'tanh', 'exp', 'log', 'sqrt', 'abs')}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}

# Short shared G/H families aligned with the screen library. Contrast may try
# every family that evaluates on the observed domain; clues only rank them.
# Not a leftover-case kernel list and not a single locked trio.
FORM_KERNEL_FAMILIES = (
    ('product', (
        {'channel': 'G', 'kernel': 'xi * xj'},
        {'channel': 'H', 'kernel': 'xi * xj * xk'},
    )),
    ('contagion', (
        {'channel': 'G', 'kernel': '(1 - xi) * xj'},
        {'channel': 'H', 'kernel': '(1 - xi) * xj * xk'},
    )),
    ('phase_diff', (
        {'channel': 'G', 'kernel': 'sin(xj - xi)'},
        {'channel': 'H', 'kernel': 'sin(xj + xk - 2 * xi)'},
    )),
    ('linear_diff', (
        {'channel': 'G', 'kernel': 'xj - xi'},
        {'channel': 'H', 'kernel': 'xj + xk - 2 * xi'},
    )),
    ('odd_source', (
        {'channel': 'G', 'kernel': 'sin(xj)'},
        {'channel': 'H', 'kernel': 'sin(xi)'},
    )),
    ('bounded_source', (
        {'channel': 'G', 'kernel': 'log(1 + exp(-xj))'},
        {'channel': 'H', 'kernel': 'xj * xk / (1 + xj + xk)'},
    )),
    ('sat_fraction', (
        {'channel': 'G', 'kernel': 'xj / (1 + xj)'},
        {'channel': 'H', 'kernel': 'xj * xk / ((1 + xj) * (1 + xk))'},
    )),
    ('exp_decay', (
        {'channel': 'G', 'kernel': 'exp(-xj)'},
        {'channel': 'H', 'kernel': 'exp(-xj - xk)'},
    )),
    ('log_growth', (
        {'channel': 'G', 'kernel': 'log(1 + xj)'},
        {'channel': 'H', 'kernel': 'log(1 + xj * xk)'},
    )),
    ('skip_middle', (
        {'channel': 'H', 'kernel': 'xi * xk'},
    )),
    ('gauss_mod', (
        {'channel': 'G', 'kernel': 'xj - xi'},
        {'channel': 'H', 'kernel': 'exp(-(xj - xk) ** 2) * (xj + xk - 2 * xi)'},
    )),
    ('lorentz_diff', (
        {'channel': 'H', 'kernel': '(xj + xk - 2 * xi) / (1 + (xj - xk) ** 2)'},
    )),
    ('tanh_phase', (
        {'channel': 'H', 'kernel': 'tanh(xj + xk - 2 * xi)'},
    )),
    ('damped_phase', (
        {'channel': 'G', 'kernel': 'sin(xj - xi) / (1 + xi ** 2)'},
        {'channel': 'H', 'kernel': 'sin(xj + xk - 2 * xi)'},
    )),
)


def form_screen_features(var: str, pairwise: str, triplet: str, pair_count: int, trip_count: int) -> list[str]:
    """Indexed features for the form families; screen may select a subset."""
    features = []
    if pair_count > 0:
        features.extend([
            f'sum[j]({pairwise}[i, j], sin({var}[j]))',
            f'sum[j]({pairwise}[i, j], tanh({var}[j]))',
            f'sum[j]({pairwise}[i, j], log(1 + exp(-{var}[j])))',
        ])
    if trip_count > 0:
        features.extend([
            f'sum[j, k]({triplet}[i, j, k], sin({var}[i]))',
            f'sum[j, k]({triplet}[i, j, k], {var}[i] * {var}[k])',
            f'sum[j, k]({triplet}[i, j, k], {var}[j] * {var}[k] / (1 + {var}[j] + {var}[k]))',
        ])
    return features


def host_state(context):
    """Keep known-edges cache on args so AgentContext._validate is not broken."""
    state = getattr(context.args, 'hypergraph_known_edges', None)
    if not isinstance(state, dict):
        state = getattr(context.args, 'known_edges', None)
    if not isinstance(state, dict):
        state = {}
    context.args.hypergraph_known_edges = state
    context.args.known_edges = state
    state.setdefault('enabled', False)
    state.setdefault('config', {})
    state.setdefault('weights', {})
    state.setdefault('observations', {})
    runtime = state.setdefault('runtime', {})
    runtime.setdefault('candidates', {})
    runtime.setdefault('gates', {})
    return state


def known_edges_enabled(context):
    for attr in ('hypergraph_known_edges', 'known_edges'):
        state = getattr(context.args, attr, None)
        if isinstance(state, dict) and bool(state.get('enabled')):
            return True
    return False


def is_multivariate(context):
    """True when the host installed more than one named scalar node variable."""
    cfg = host_state(context).get('config') or {}
    try:
        return int(cfg.get('state_dim') or 1) > 1
    except (TypeError, ValueError):
        return False


is_vector = is_multivariate


def sync_relation_weights(context, weights=None):
    """Install `{relation}_weights` so engine sums use GPT-style edge intensity.

    Formula text stays `sum[j](pairwise[i,j], kernel)` with kernel coefficient 1.
    Missing weights mean 1. `install_relations` must be followed by this call
    because it rebuilds the integer tables without the weight fields.
    """
    if weights is None:
        weights = host_state(context).get('weights') or {}
    drop = [f'{name}_weights' for name in context.relation_names]
    for key in drop:
        context.data.pop(key, None)
        context.variable_descriptions.pop(key, None)
        context.variable_axes.pop(key, None)
    structures = {
        key: rel for key, rel in context.variable_structures.items() if key not in drop
    }
    for name, raw in dict(weights).items():
        if name not in context.relation_names:
            continue
        rows = np.asarray(context.data[name])
        array = np.asarray(raw, float).reshape(-1)
        if array.shape != (len(rows),):
            raise ValueError(f'{name}_weights length {array.shape} != rows {len(rows)}')
        key = f'{name}_weights'
        context.data[key] = array
        context.variable_descriptions[key] = (
            f'Row-aligned weights for {name}; kernel stays unweighted, intensity on the edge.'
        )
        context.variable_axes[key] = ()
        structures[key] = name
    context.variable_structures = structures
    context.invalidate_splits()
    context._validate()


def kernel(text, values):
    """Small scalar AST interpreter; never eval/exec caller text."""
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError('kernel must be a string of <=1000 characters')
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 160:
        raise ValueError('kernel too complex')
    def run(n):
        if isinstance(n, ast.Expression): return run(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float): return n.value
        if isinstance(n, ast.Name) and n.id in values: return values[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](run(n.left), run(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -run(n.operand) if isinstance(n.op, ast.USub) else run(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUN and len(n.args) == 1 and not n.keywords:
            return FUN[n.func.id](run(n.args[0]))
        raise ValueError('unsupported kernel syntax or variable')
    with np.errstate(all='raise'):
        return np.asarray(run(tree), dtype=float)


def inputs(context):
    d = context.data
    x, y = np.asarray(d['x'], float), np.asarray(d['dx_dt'], float)
    if x.ndim != 3 or x.shape != y.shape or x.shape[-1] != context.num_nodes:
        raise ValueError('STOP: require univariate x, dx_dt as (trajectory,time,node); never flatten named variables into the node axis')
    if min(x.shape) < 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid/nonfinite observations')
    return x, y


def relations(context):
    cfg = host_state(context)['config']
    result = {}
    for channel, arity, key in [('G', 2, 'pairwise'), ('H', 3, 'triplet')]:
        names = [n for n in context.relation_names if np.asarray(context.data[n]).ndim == 2 and np.asarray(context.data[n]).shape[1] == arity]
        name = cfg.get(key)
        if name is None:
            if len(names) != 1: raise ValueError('supply exactly one installed relation of each arity, including empty tables')
            name = names[0]
        if name not in names: raise ValueError('configured relation arity/name mismatch')
        raw = np.asarray(context.data[name])
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all(): raise ValueError('noninteger node index')
        rows = raw.astype(int)
        if rows.size and (rows.min() < 0 or rows.max() >= context.num_nodes): raise ValueError('node index out of range')
        if any(len(set(r)) != arity for r in rows): raise ValueError('repeated endpoint')
        if len(set(map(tuple, rows))) != len(rows): raise ValueError('duplicate relation rows: clarify host counting')
        stored = host_state(context)['weights'].get(name)
        if stored is None:
            stored = context.data.get(name + '_weights', np.ones(len(rows)))
        weights = np.asarray(stored, float)
        if weights.shape != (len(rows),) or not np.isfinite(weights).all(): raise ValueError('invalid row weights')
        factor = 1.0
        convention = cfg.get('triplet_counting', 'rows')
        if convention not in ('rows', 'ordered_symmetric'): raise ValueError('unknown triplet counting')
        if channel == 'H' and convention == 'ordered_symmetric':
            lookup = {tuple(r): w for r, w in zip(rows, weights)}
            for (i,j,k), w in lookup.items():
                if (i,k,j) not in lookup or lookup[i,k,j] != w: raise ValueError('ordered_symmetric requires matched swaps and equal weights')
            factor = 0.5
        result[channel] = (name, rows, weights * factor)
    return result


def signature(context):
    if is_multivariate(context):
        from ._hypergraph_multivariate import multivariate_signature
        return multivariate_signature(context)
    h = hashlib.sha256()
    x,y = inputs(context)
    for a in (x,y): h.update(str(a.shape).encode()); h.update(a.tobytes())
    for key,(name,rows,w) in relations(context).items():
        h.update((key+name).encode()); h.update(rows.tobytes()); h.update(w.tobytes())
    obs = host_state(context)['observations'] or context.data.get('_known_edges_observations', {})
    for k in ('x','t'):
        if k in obs:
            a=np.asarray(obs[k],float); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def column(context, x, term):
    channel, text = term['channel'], term['kernel']
    if channel == 'F':
        out=np.broadcast_to(kernel(text, {'xi':x}), x.shape)
        if not np.isfinite(out).all(): raise ValueError('nonfinite local kernel')
        return out
    if channel not in ('G','H'): raise ValueError('channel must be F/G/H')
    _,rows,weights = relations(context)[channel]
    # Empty channels are conventions, never inferred kernels.
    if not len(rows): return np.zeros_like(x)
    out = np.zeros_like(x)
    for row,w in zip(rows,weights):
        vals = {'xi':x[...,row[0]],'xj':x[...,row[1]]}
        if channel == 'H': vals['xk'] = x[...,row[2]]
        out[...,row[0]] += w * kernel(text,vals)
    if not np.isfinite(out).all(): raise ValueError('kernel domain failure')
    return out


def normalize_terms(context, terms):
    if len(terms)>24: raise ValueError('at most 24 terms; use a small hypothesis contrast')
    out=[]
    for t in terms:
        q={'channel':t['channel'],'kernel':t['kernel']}
        if q['channel'] in ('G','H') and not len(relations(context)[q['channel']][1]): continue
        # Check syntax/domain now; no silent invalid-column dropping.
        column(context, inputs(context)[0], q)
        if q not in out: out.append(q)
    return out


def design(context, terms, mode='none'):
    x,y=inputs(context)
    cols=[column(context,x,t).ravel() for t in terms]
    if mode not in ('none','constant','linear'): raise ValueError('only node constant OR node slope allowed')
    if mode!='none':
        for i in range(x.shape[-1]):
            a=np.zeros_like(x); a[...,i]=1 if mode=='constant' else x[...,i]; cols.append(a.ravel())
    return (np.column_stack(cols) if cols else np.zeros((y.size,0))), y.ravel()


def solve(A,y):
    if not A.shape[1]: return np.zeros(0),0,None
    scale=np.sqrt(np.mean(A*A,axis=0))
    if np.any(scale<=1e-15): raise ValueError('unexcited/zero design column: remove term or stop')
    B=A/scale
    c,_,rank,s=np.linalg.lstsq(B,y,rcond=None)
    return c/scale,int(rank),float(s[0]/s[-1]) if s[-1]>0 else None


def metrics(context, terms, coefficients, mode='none'):
    A,y=design(context,terms,mode); c=np.asarray(coefficients,float)
    if c.shape!=(A.shape[1],) or not np.isfinite(c).all(): raise ValueError('invalid coefficients')
    x,_=inputs(context); residual=(y-A@c).reshape(x.shape)
    rms=float(np.sqrt(np.mean(residual**2))); scale=float(np.sqrt(np.mean(y*y)))
    _,rank,cond=solve(A,y)
    return {'derivative_rmse':rms,'derivative_relative_rmse':rms/max(scale,1e-15),
            'derivative_rms':scale,'per_node_rmse':np.sqrt(np.mean(residual**2,axis=(0,1))).tolist(),
            'per_trajectory_rmse':np.sqrt(np.mean(residual**2,axis=(1,2))).tolist(),
            'rank':rank,'columns':A.shape[1],'condition_number':cond}


def fit(context,terms,mode='none'):
    A,y=design(context,terms,mode); c,_,_=solve(A,y)
    return c,metrics(context,terms,c,mode)


def _node_row_mask(context, node: int):
    x, _ = inputs(context)
    n_node = int(x.shape[-1])
    return (np.arange(x.size) % n_node) == int(node)


def _split_clean_dirty(per_node_rmse, deriv_rms: float):
    rmse = np.asarray(per_node_rmse, dtype=float)
    tol = max(1e-4, 5e-4 * float(deriv_rms or 1.0))
    dirty_cut = max(1e-2, 0.05 * float(deriv_rms or 1.0))
    return rmse <= tol, rmse >= dirty_cut


def consensus_shared_coefficients(context, terms, global_c, *, rmse_tol: float = None):
    """If >=2 nodes already match one short shared kernel, lock that amplitude.

    Dirty nodes are not allowed to pull coefficients. A true graph has no dirty
    split, so this returns None and leaves the global fit.
    """
    if not terms:
        return None
    x, y_full = inputs(context)
    deriv_rms = float(np.sqrt(np.mean(np.asarray(y_full) ** 2)))
    A, y = design(context, terms, 'none')
    n_node = int(x.shape[-1])
    if n_node < 3 or A.shape[1] == 0:
        return None
    global_c = np.asarray(global_c[: len(terms)], dtype=float)
    probes = [global_c]
    ones = np.array([0.0 if t['channel'] == 'F' else 1.0 for t in terms], dtype=float)
    probes.append(ones)
    probes.append(np.round(global_c, 3))
    best = None
    for probe in probes:
        try:
            m = metrics(context, terms, probe, 'none')
        except (ValueError, np.linalg.LinAlgError):
            continue
        clean, dirty = _split_clean_dirty(m['per_node_rmse'], deriv_rms)
        n_clean = int(np.sum(clean))
        if n_clean >= 2 and np.any(dirty):
            if best is None or n_clean > best[0]:
                best = (n_clean, np.asarray(probe, dtype=float))
    clean_c = []
    for node in range(n_node):
        keep = _node_row_mask(context, node)
        if int(keep.sum()) <= A.shape[1]:
            continue
        try:
            ci, rank, _ = solve(A[keep], y[keep])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if rank != A.shape[1] or not np.isfinite(ci).all():
            continue
        residual = y[keep] - A[keep] @ ci
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        local_tol = max(1e-4, 5e-4 * deriv_rms) if rmse_tol is None else rmse_tol
        if rmse <= local_tol:
            clean_c.append(ci)
    if len(clean_c) >= 2:
        stacked = np.vstack(clean_c)
        locked = np.median(stacked, axis=0)
        spread = np.max(np.std(stacked, axis=0))
        scale = max(float(np.max(np.abs(locked))), 1.0)
        if spread <= 0.05 * scale:
            m = metrics(context, terms, locked, 'none')
            clean, dirty = _split_clean_dirty(m['per_node_rmse'], deriv_rms)
            n_clean = int(np.sum(clean))
            if n_clean >= 2 and np.any(dirty) and (best is None or n_clean >= best[0]):
                best = (n_clean, locked)
    if best is None:
        return None
    locked = best[1]
    if np.allclose(locked, global_c, rtol=1e-4, atol=1e-4):
        return None
    return locked


def subset_exact_ok(context, terms, coefficients, mode='none', *, rmse_tol: float = None) -> bool:
    if mode != 'none':
        return False
    m = metrics(context, terms, coefficients, mode)
    _, y = inputs(context)
    deriv_rms = float(np.sqrt(np.mean(np.asarray(y) ** 2)))
    clean, dirty = _split_clean_dirty(m['per_node_rmse'], deriv_rms)
    return int(np.sum(clean)) >= 2 and bool(np.any(dirty))


def crosscheck(context,terms,mode='none'):
    A,y=design(context,terms,mode); x,_=inputs(context)
    if len(x)<2: return {'available':False,'reason':'need >=2 trajectories'}
    ids=np.repeat(np.arange(len(x)),x.shape[1]*x.shape[2]); coeff=[]; errors=[]; ranks=[]
    for i in range(len(x)):
        c,r,_=solve(A[ids!=i],y[ids!=i]); coeff.append(c); ranks.append(r)
        errors.append(float(np.sqrt(np.mean((y[ids==i]-A[ids==i]@c)**2))))
    c=np.asarray(coeff)
    return {'available':True,'heldout_rmse':errors,'max_heldout_rmse':max(errors),
            'full_rank':all(r==A.shape[1] for r in ranks),
            'coefficient_spread':np.ptp(c,axis=0).tolist()}


def registry(context):
    return host_state(context)['runtime']


def save(context,terms,c,mode='none',stage='fit',extra=None):
    candidate={'terms':terms,'coefficients':np.asarray(c,float).tolist(),'node_mode':mode,
               'data_signature':signature(context),'stage':stage}
    if extra: candidate.update(extra)
    cid=hashlib.sha256(json.dumps(candidate,sort_keys=True,allow_nan=False).encode()).hexdigest()[:20]
    registry(context)['candidates'][cid]=copy.deepcopy(candidate)
    return cid


def get(context,cid):
    c=copy.deepcopy(registry(context)['candidates'][cid])
    actual=hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()[:20]
    if actual!=cid: raise ValueError('candidate content changed after registration')
    if c['data_signature']!=signature(context): raise ValueError('stale candidate: observations/weights/edges changed')
    return c


def summary(context,cid,next_action,**extra):
    c=get(context,cid); terms=c['terms']; coef=c['coefficients']; m=metrics(context,terms,coef,c['node_mode'])
    channels={k:[] for k in ('F','G','H')}
    for t,a in zip(terms,coef): channels[t['channel']].append(f'({a:.12g})*({t["kernel"]})')
    channels={k:' + '.join(v) or '0' for k,v in channels.items()}
    if c['node_mode']!='none': channels['F'] += ' + node_parameter[i]' + ('*xi' if c['node_mode']=='linear' else '')
    node=np.asarray(coef[len(terms):])
    return {'candidate_id':cid,'channels':channels,'node_mode':c['node_mode'],
            'node_parameter_range':[float(node.min()),float(node.max())] if node.size else None,
            'metrics':m,'can_submit':False,'symbolic_equivalence_proven':False,
            'next_action':next_action,**extra}


def readable(result):
    return json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)


def screen_terms(coefficients,pairwise,triplet):
    terms=[]; cs=[]
    for text,c in coefficients.items():
        if text=='bias': ch,k='F','1'
        elif text.strip().startswith('sum['):
            match=re.match(r'^sum\[[^\]]+\]\(\s*(\w+)\[[^\]]+\]\s*,\s*(.*)\)$',text.strip())
            if not match: raise ValueError('unrecognized screen reduction; explicit adapter needed')
            name,k=match.groups(); ch={pairwise:'G',triplet:'H'}[name]
        else: ch,k='F',text
        for idx,var in [('i','xi'),('j','xj'),('k','xk')]: k=re.sub(r'x\[\s*'+idx+r'\s*\]',var,k)
        terms.append({'channel':ch,'kernel':k}); cs.append(c)
    return terms,cs


def rhs(context,candidate,x):
    out=np.zeros_like(x,dtype=float); terms=candidate['terms']; cs=candidate['coefficients']
    for t,c in zip(terms,cs): out+=c*column(context,x,t)
    node=np.asarray(cs[len(terms):])
    if candidate['node_mode']=='constant': out+=node
    if candidate['node_mode']=='linear': out+=node*x
    return out


def clamp_refine_budget(coefficient_cutoff=1e-4, contribution_cutoff=1e-4,
                        residual_allowance=1e-8, round_digits=4, max_denominator=32, **_ignored):
    """Clip caller-supplied refine knobs; never abort the call for an out-of-range slack."""
    def _clip(value, default, lo, hi, cast):
        try:
            number = cast(value)
        except (TypeError, ValueError):
            number = cast(default)
        if isinstance(number, float) and number != number:
            number = cast(default)
        return cast(min(hi, max(lo, number)))
    return {
        'coefficient_cutoff': _clip(coefficient_cutoff, 1e-4, 0.0, 0.01, float),
        'contribution_cutoff': _clip(contribution_cutoff, 1e-4, 0.0, 0.01, float),
        'residual_allowance': _clip(residual_allowance, 1e-8, 0.0, 1e-5, float),
        'round_digits': _clip(round_digits, 4, 0, 10, int),
        'max_denominator': _clip(max_denominator, 32, 1, 100, int),
    }


def near_simple_round(old, new):
    """Fitted amplitude is already the simple number within discretization noise."""
    old, new = float(old), float(new)
    return abs(new - old) <= max(1e-4, abs(old) * 1e-4)


def candidate_complexity(candidate: dict) -> int:
    extra = 0 if candidate.get('node_mode') in (None, 'none') else 1
    if candidate.get('kind') in ('multivariate', 'vector'):
        blocks = candidate.get('per_variable') or candidate.get('per_component') or {}
        n = 0
        for block in blocks.values():
            n += len(block.get('terms') or [])
        return n + extra
    return len(candidate.get('terms') or []) + extra


def _gate_rmse(gate: dict) -> float:
    try:
        value = float(gate.get('resim_rmse'))
    except (TypeError, ValueError):
        return float('inf')
    return value if value == value else float('inf')


def select_best_gated_id(context):
    """Exact (RMSE<=1e-5) wins; otherwise shortest reintegrated candidate, then lowest RMSE."""
    runtime = registry(context)
    ranked = []
    for cid, gate in runtime['gates'].items():
        if gate.get('status') == 'blocked' or cid not in runtime['candidates']:
            continue
        exact = bool(gate.get('can_submit') or gate.get('status') == 'passed')
        ranked.append((0 if exact else 1, candidate_complexity(runtime['candidates'][cid]), _gate_rmse(gate), cid))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][-1]


def reintegrate_next_action(candidate_id, *, passed: bool) -> str:
    if passed:
        return f'submit_formula(f={candidate_id})'
    return (
        'if another rollback/family remains, hypergraph_formula_refine then reintegrate; '
        'if some nodes already match a short kernel to derivative precision, lock that kernel '
        'and do not retune amplitude on the remaining nodes; '
        'if the search budget is exhausted, submit_formula on the shortest already-reintegrated '
        'candidate_id (tie-break: lower resim RMSE). exact_gate stays failed'
    )


def require_submission(context,cid):
    """Host MUST call immediately before submit/auto-candidate acceptance.
    Returns the exact gated candidate; serialization to engine syntax is a host
    adapter, not a made-up engine API. Never accept arbitrary LLM formula text.

    Exact recovery still needs can_submit (RMSE<=1e-5). If reintegration ran and
    failed, the same ID may be submitted as best-effort so a kernel is always
    produced when the given graph cannot close the ODE.
    """
    runtime = registry(context)
    if cid not in runtime['candidates']:
        gated = [i for i, gate in runtime['gates'].items() if gate.get('status') != 'blocked']
        hint = f'; use candidate_id {gated[-1]} from hypergraph_reintegrate' if gated else ''
        raise ValueError(
            'submission blocked: f must be the exact gated candidate_id from hypergraph_reintegrate, '
            'not a component name or a handwritten formula' + hint
        )
    c=get(context,cid); gate=runtime['gates'].get(cid,{})
    if gate.get('data_signature') and gate.get('data_signature') != signature(context):
        raise ValueError('submission blocked: stale reintegration gate; re-run hypergraph_reintegrate')
    if gate.get('status') == 'blocked':
        raise ValueError('submission blocked: reintegration is blocked; host must supply the original grid')
    if not gate:
        raise ValueError('submission blocked: this candidate_id has no reintegration gate; call hypergraph_reintegrate first')
    if not gate.get('can_submit') and gate.get('status') == 'passed':
        raise ValueError(
            f"submission blocked: reintegration passed (rmse={gate.get('resim_rmse')}) but can_submit is false"
        )
    return c


def weighted_screen(tool, features, pairwise, triplet, max_terms):
    """Reuse the existing BIC-Lasso selector on weighted columns; no new selector.
    Internal _bic_lasso_selection is visible in the supplied graph_analysis.py.
    Host training split must retain node axis, as original graph tools require.
    """
    from .graph_analysis import GraphFeatureFitTool
    terms,_=screen_terms(dict.fromkeys(features,1.0),pairwise,triplet)
    train=tool.context.train_split
    x=np.asarray(train.data['x'],float); y=np.asarray(train.data['dx_dt'],float)
    if x.shape!=y.shape or x.shape[-1]!=tool.context.num_nodes: raise ValueError('training split must preserve node axis')
    columns=[]; valid=[]; texts=[]; rejected=[]
    for t,text in zip(terms,features):
        try:
            col=column(tool.context,x,t).ravel()
            if not np.isfinite(col).all(): raise ValueError('nonfinite column')
            columns.append(col); valid.append(t); texts.append(text)
        except (ValueError,FloatingPointError,OverflowError) as exc:
            rejected.append({'feature':text,'reason':str(exc)})
    if not columns: raise ValueError('no finite library columns')
    A=np.column_stack(columns)
    selected,trace,beam=GraphFeatureFitTool._bic_lasso_selection(
        A,y.ravel(),[tool.parse_formula(s) for s in texts],
        include_bias=True,max_terms=max(1,min(int(max_terms),20)))
    D=np.column_stack([np.ones(y.size),A[:,selected]])
    coef=np.linalg.lstsq(D,y.ravel(),rcond=None)[0]
    return {'fitted_coefficients':{'bias':float(coef[0]),**{texts[j]:float(a) for j,a in zip(selected,coef[1:])}},
            'rejected_features':rejected,'selection_trace':trace,'beam_trace':beam,
            'weighted_screen':True,'is_candidate':False}


def screen_handoff(context, screen, pairwise, triplet):
    terms,c=screen_terms(screen['fitted_coefficients'],pairwise,triplet)
    terms_and_c=[(t,a) for t,a in zip(terms,c) if a!=0 and
                 (t['channel']=='F' or len(relations(context)[t['channel']][1]))]
    terms=[t for t,_ in terms_and_c]; c=[a for _,a in terms_and_c]
    cid=save(context,terms,c,stage='screen')
    return summary(context,cid,'hypergraph_formula_refine; if persistent node clues then hypergraph_node_parameter_fit')


def attach_node_parameter(context, candidate):
    """Install a node-aligned feature so the engine can evaluate omega[i] / mu[i]*x[i]."""
    mode = candidate['node_mode']
    if mode == 'none':
        return None
    terms = candidate['terms']
    node = np.asarray(candidate['coefficients'][len(terms):], float)
    n = int(context.num_nodes)
    if node.shape != (n,) or not np.isfinite(node).all():
        raise ValueError('node parameter vector does not match num_nodes')
    name = 'omega' if mode == 'constant' else 'mu'
    x = np.asarray(context.data['x'])
    feature = np.broadcast_to(node.reshape((1,) * (x.ndim - 1) + (n,)), x.shape).copy()
    context.data[name] = feature
    context.variable_descriptions[name] = 'Per-node parameter; last axis is the node index.'
    context.variable_axes[name] = ()
    context.invalidate_splits()
    host_state(context)['exported_node'] = {
        'name': name,
        'values': node.tolist(),
        'mode': mode,
    }
    return name


def serialize_indexed(context, candidate):
    """Host adapter: turn a gated F/G/H candidate into engine indexed syntax."""
    if candidate.get('kind') in ('multivariate', 'vector'):
        from ._hypergraph_multivariate import serialize_multivariate
        return serialize_multivariate(context, candidate)
    rel = relations(context)
    pair, trip = rel['G'][0], rel['H'][0]
    parts = []
    name = attach_node_parameter(context, candidate)
    if name == 'omega':
        parts.append('omega[i]')
    elif name == 'mu':
        parts.append('mu[i] * x[i]')
    terms = candidate['terms']
    coefs = candidate['coefficients']
    for term, coef in zip(terms, coefs):
        kernel_text = term['kernel']
        for src, indexed in (('xk', 'x[k]'), ('xj', 'x[j]'), ('xi', 'x[i]')):
            kernel_text = re.sub(r'\b' + src + r'\b', indexed, kernel_text)
        if term['channel'] == 'F':
            parts.append(f'({coef:.12g}) * ({kernel_text})')
        elif term['channel'] == 'G':
            parts.append(f'({coef:.12g}) * sum[j]({pair}[i, j], {kernel_text})')
        elif term['channel'] == 'H':
            parts.append(f'({coef:.12g}) * sum[j, k]({trip}[i, j, k], {kernel_text})')
        else:
            raise ValueError('channel must be F/G/H')
    return ' + '.join(parts) if parts else '0'


def json_safe(value):
    """Original fitter traces may include inf BIC; expose unavailable as null."""
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value
