"""Multivariate known-edges branch: named scalar node variables u, v, …."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp

from ._hypergraph_known_edge_common import (
    clamp_refine_budget, get, host_state, json_safe, kernel, near_simple_round,
    readable, registry, reintegrate_next_action, relations, solve,
)


def variable_blocks(cand):
    blocks = cand.get('per_variable')
    if blocks is None:
        blocks = cand.get('per_component')
    return {} if blocks is None else blocks

# Catalog describes structure, not a particular state dimension or system.
MULTIVARIATE_FORM_KERNEL_FAMILIES = (
    'local_linear_cross', 'local_bilinear_cross',
    'linear_diff', 'phase_diff', 'product',
    'cubic_diff', 'cubic_ordered', 'cubic_pair_sum',
)

# Generic coupling contrast, shared with the vector screen library.
# Degree-3 families are refine-only: never added to _coupling_library().
MULTIVARIATE_COUPLING_FAMILIES = (
    ('linear_diff', (
        {'channel': 'G', 'kernel': 'uj - ui'},
        {'channel': 'H', 'kernel': 'uj + uk - 2 * ui'},
    )),
    ('phase_diff', (
        {'channel': 'G', 'kernel': 'sin(uj - ui)'},
        {'channel': 'H', 'kernel': 'sin(uj + uk - 2 * ui)'},
    )),
    ('product', (
        {'channel': 'G', 'kernel': 'ui * uj'},
        {'channel': 'H', 'kernel': 'ui * uj * uk'},
    )),
)

# Single-channel cubic coupling. Tried only when refine contrast is on
# (large residual / long soup). Pairwise diffusion may be kept if the family
# has no G of its own. Not a named leftover system.
MULTIVARIATE_POLY3_FAMILIES = (
    ('cubic_diff', (
        {'channel': 'G', 'kernel': 'uj**3 - ui**3'},
        {'channel': 'H', 'kernel': 'uj**3 + uk**3 - 2 * ui**3'},
    )),
    ('cubic_ordered', (
        {'channel': 'G', 'kernel': 'uj - ui'},
        {'channel': 'H', 'kernel': 'uj * uk**2'},
        {'channel': 'H', 'kernel': 'uk * uj**2'},
        {'channel': 'H', 'kernel': 'ui**3'},
    )),
    ('cubic_pair_sum', (
        {'channel': 'G', 'kernel': 'uj - ui'},
        {'channel': 'H', 'kernel': 'uj * uk * (uj + uk)'},
        {'channel': 'H', 'kernel': 'ui**3'},
    )),
)


def component_names(context):
    cfg = host_state(context)['config']
    names = list(cfg.get('components') or [])
    if not names:
        raise ValueError('multivariate host must set config.components to named scalar variables')
    return names


def vector_fields(context):
    names = component_names(context)
    fields = {}
    shape = None
    for name in names:
        arr = np.asarray(context.data[name], float)
        if arr.ndim != 3 or arr.shape[-1] != context.num_nodes:
            raise ValueError(f'node variable {name} must be (trajectory,time,node)')
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError('named node variables have mismatched shapes')
        if not np.isfinite(arr).all():
            raise ValueError(f'nonfinite vector field {name}')
        fields[name] = arr
    return fields


def vector_targets(context):
    fields = vector_fields(context)
    out = {}
    for name in fields:
        y = np.asarray(context.data[f'd{name}_dt'], float)
        if y.shape != fields[name].shape or not np.isfinite(y).all():
            raise ValueError(f'vector target d{name}_dt must match {name}')
        out[name] = y
    return fields, out


def multivariate_signature(context):
    h = hashlib.sha256()
    fields, targets = vector_targets(context)
    for name in fields:
        for a in (fields[name], targets[name]):
            h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    for key, (rel_name, rows, w) in relations(context).items():
        h.update((key + rel_name).encode()); h.update(rows.tobytes()); h.update(w.tobytes())
    obs = host_state(context)['observations']
    for k in ('x', 't'):
        if k in obs:
            a = np.asarray(obs[k], float)
            h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _broadcast_kernel(text, vals, shape):
    col = kernel(text, vals)
    col = np.asarray(col, float)
    if col.shape == shape:
        return col
    return np.broadcast_to(col, shape).copy()


def vector_column(context, term, fields=None):
    fields = fields if fields is not None else vector_fields(context)
    ref = next(iter(fields.values()))
    channel, text = term['channel'], term['kernel']
    if channel == 'F':
        vals = {name + 'i': arr for name, arr in fields.items()}
        return _broadcast_kernel(text, vals, ref.shape)
    _name, rows, weights = relations(context)[channel]
    out = np.zeros_like(ref)
    if not len(rows):
        return out
    for row, weight in zip(rows, weights):
        vals = {}
        for name, arr in fields.items():
            vals[name + 'i'] = arr[..., row[0]]
            vals[name + 'j'] = arr[..., row[1]]
            if channel == 'H':
                vals[name + 'k'] = arr[..., row[2]]
        out[..., row[0]] += weight * kernel(text, vals)
    if not np.isfinite(out).all():
        raise ValueError('vector kernel domain failure')
    return out


def normalize_vector_terms(context, terms):
    out = []
    for term in terms:
        rec = {'channel': term['channel'], 'kernel': term['kernel']}
        if rec['channel'] not in ('F', 'G', 'H'):
            raise ValueError('invalid channel')
        if rec['channel'] in ('G', 'H') and not len(relations(context)[rec['channel']][1]):
            continue
        if not np.isfinite(vector_column(context, rec)).all():
            raise ValueError('nonfinite kernel')
        if rec not in out:
            out.append(rec)
    return out


def vector_design(context, terms, target, node_mode='none'):
    fields, targets = vector_targets(context)
    y = targets[target].ravel()
    cols = [vector_column(context, term, fields).ravel() for term in terms]
    if node_mode in ('constant', 'linear'):
        x0 = fields[target]
        n = x0.shape[-1]
        for i in range(n):
            a = np.zeros_like(x0)
            a[..., i] = 1.0 if node_mode == 'constant' else x0[..., i]
            cols.append(a.ravel())
    elif node_mode != 'none':
        raise ValueError('vector node_mode supports none, constant, or linear')
    A = np.column_stack(cols) if cols else np.zeros((y.size, 0))
    return A, y


def vector_metrics(context, terms, coefficients, target, node_mode='none'):
    A, y = vector_design(context, terms, target, node_mode)
    c = np.asarray(coefficients, float)
    residual = (y - A @ c)
    fields = vector_fields(context)
    shape = next(iter(fields.values())).shape
    rms = float(np.sqrt(np.mean(residual ** 2)))
    scale = float(np.sqrt(np.mean(y * y)))
    _, rank, cond = solve(A, y)
    return {
        'derivative_rmse': rms,
        'derivative_relative_rmse': rms / max(scale, 1e-15),
        'rank': rank,
        'columns': A.shape[1],
        'condition_number': cond,
        'residual': residual.reshape(shape),
    }


def vector_fit(context, terms, target, node_mode='none'):
    A, y = vector_design(context, terms, target, node_mode)
    c, _, _ = solve(A, y)
    return c, vector_metrics(context, terms, c, target, node_mode)


def vector_crosscheck(context, terms, target, node_mode='none'):
    A, y = vector_design(context, terms, target, node_mode)
    fields = vector_fields(context)
    x = next(iter(fields.values()))
    if len(x) < 2:
        return {'available': False}
    ids = np.repeat(np.arange(len(x)), x.shape[1] * x.shape[2])
    errors = []
    ranks = []
    coefficients = []
    for i in range(len(x)):
        c, r, _ = solve(A[ids != i], y[ids != i])
        ranks.append(r)
        coefficients.append(c)
        errors.append(float(np.sqrt(np.mean((y[ids == i] - A[ids == i] @ c) ** 2))))
    return {
        'available': True,
        'heldout_rmse': errors,
        'max_heldout_rmse': max(errors),
        'full_rank': all(r == A.shape[1] for r in ranks),
        'coefficient_spread': np.ptp(coefficients, axis=0).tolist(),
    }


def multivariate_save(context, per_component, stage='fit', extra=None):
    blocks = copy.deepcopy(per_component)
    candidate = {
        'kind': 'multivariate',
        'terms': [],
        'coefficients': [],
        'node_mode': 'none',
        'components': component_names(context),
        'per_variable': blocks,
        'per_component': blocks,
        'data_signature': multivariate_signature(context),
        'stage': stage,
    }
    if extra:
        candidate.update(extra)
    cid = hashlib.sha256(json.dumps(candidate, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    registry(context)['candidates'][cid] = copy.deepcopy(candidate)
    return cid


def _local_f_library(names):
    terms = [{'channel': 'F', 'kernel': '1'}]
    for name in names:
        terms.append({'channel': 'F', 'kernel': f'{name}i'})
    for i, a in enumerate(names):
        for b in names[i:]:
            terms.append({'channel': 'F', 'kernel': f'{a}i * {b}i'})
    return terms


def _coupling_library():
    return [
        {'channel': 'G', 'kernel': 'uj - ui'},
        {'channel': 'G', 'kernel': 'sin(uj - ui)'},
        {'channel': 'G', 'kernel': 'ui * uj'},
        {'channel': 'H', 'kernel': 'uj + uk - 2 * ui'},
        {'channel': 'H', 'kernel': 'sin(uj + uk - 2 * ui)'},
        {'channel': 'H', 'kernel': 'ui * uj * uk'},
    ]


def _screen_component(context, target, max_terms):
    names = component_names(context)
    library = _local_f_library(names) + [dict(channel=t['channel'], kernel=re.sub(
        r'\bu([ijk])\b', lambda m: names[0] + m[1], t['kernel'])) for t in _coupling_library()]
    usable = []
    columns = []
    for term in library:
        try:
            recs = normalize_vector_terms(context, [term])
            if not recs:
                continue
            col = vector_column(context, recs[0]).ravel()
            if not np.isfinite(col).all() or float(np.sqrt(np.mean(col * col))) <= 1e-15:
                continue
            usable.append(recs[0])
            columns.append(col)
        except (ValueError, FloatingPointError, OverflowError):
            continue
    if not columns:
        raise ValueError(f'no finite vector library columns for {target}')
    A = np.column_stack(columns)
    y = vector_targets(context)[1][target].ravel()
    from .graph_analysis import GraphFeatureFitTool
    selected, trace, _beam = GraphFeatureFitTool._bic_lasso_selection(
        A, y, [], include_bias=False, max_terms=max(1, min(int(max_terms), 12)),
    )
    terms = [usable[j] for j in selected]
    coef, metrics = vector_fit(context, terms, target)
    return terms, coef, metrics, trace


def diagnose_multivariate(context, candidate_id=''):
    fields, targets = vector_targets(context)
    rel = relations(context)
    names = component_names(context)
    per = {}
    for name in names:
        y = targets[name]
        per[name] = {
            'derivative_rms': float(np.sqrt(np.mean(y * y))),
            'state_range': [float(fields[name].min()), float(fields[name].max())],
        }
    residual_rmse = None
    if candidate_id:
        cand = get(context, candidate_id)
        residual_rmse = float(np.sqrt(np.mean(vector_full_residual(context, cand) ** 2)))
        for name, block in variable_blocks(cand).items():
            per[name].update(_component_report(context, name, block))
            per[name]['recommended_node_mode'] = 'none'
            if block.get('node_mode', 'none') == 'none':
                for mode in ('constant', 'linear'):
                    if all(_node_attempt(context, block['terms'], name, mode)[-1].values()):
                        per[name]['recommended_node_mode'] = mode
                        break
    incoming = np.zeros(next(iter(fields.values())).shape[-1], int)
    for ch in ('G', 'H'):
        if len(rel[ch][1]):
            np.add.at(incoming, rel[ch][1][:, 0], 1)
    return json_safe({
        'status': 'ok',
        'multivariate': True,
        'vector': True,
        'state_dim': len(names),
        'components': names,
        'per_variable': per,
        'per_component': per,
        'shape': list(next(iter(fields.values())).shape),
        'residual_rmse': residual_rmse,
        'channels': {
            ch: {'relation': rel[ch][0], 'rows': len(rel[ch][1]), 'empty_convention': not len(rel[ch][1])}
            for ch in ('G', 'H')
        },
        'isolated_target_nodes': np.flatnonzero(incoming == 0).tolist(),
        'clues_are_not_authorization': True,
        'can_submit': False,
        'next_action': (
            ('read per_variable recommended_node_mode; hypergraph_node_parameter_fit only with evidence, otherwise hypergraph_formula_refine or hypergraph_reintegrate. ' if candidate_id else 'hypergraph_formula_screen. ') +
            'F may use all local named variables; '
            'G/H may be identically 0 on a variable. Do not flatten variables into nodes.'
        ),
    })


def screen_multivariate(context, max_terms=6):
    names = component_names(context)
    per = {}
    channels = {}
    for name in names:
        terms, coef, metrics, _trace = _screen_component(context, name, max_terms)
        per[name] = {
            'terms': terms,
            'coefficients': np.asarray(coef, float).tolist(),
            'node_mode': 'none',
        }
        channels[name] = {
            'F': _format_channel(terms, coef, 'F'),
            'G': _format_channel(terms, coef, 'G'),
            'H': _format_channel(terms, coef, 'H'),
            'derivative_rmse': metrics['derivative_rmse'],
        }
    cid = multivariate_save(context, per, 'screen')
    return json_safe({
        **multivariate_summary(context, cid, 'hypergraph_formula_refine'),
        'formula': serialize_multivariate(context, get(context, cid)),
        'known_edges': {
            'candidate_id': cid,
            'multivariate': True,
            'vector': True,
            'components': channels,
            'can_submit': False,
        },
        'candidate_id': cid,
        'can_submit': False,
        'screen_configuration': {'multivariate': True, 'vector': True, 'components': names, 'max_terms': max_terms},
    })


def _format_channel(terms, coef, channel):
    parts = []
    for term, amp in zip(terms, np.asarray(coef, float)):
        if term['channel'] == channel:
            parts.append(f'({amp:.12g})*({term["kernel"]})')
    return ' + '.join(parts) or '0'


def _family_trials(context, terms):
    names = component_names(context)
    local = [t for t in terms if t in _local_f_library(names)]
    coupling = [t for t in terms if t['channel'] != 'F']
    # Local cross-component repairs retain the current short coupling support.
    for family, basis in (
        ('local_linear_cross', _local_f_library(names)[:1 + len(names)]),
        ('local_bilinear_cross', _local_f_library(names)),
    ):
        if len(coupling) <= 3:
            yield family, basis + coupling
    # Input component and output component are independent. Never merge families.
    def _map_family(records):
        for source in names:
            mapped = [{'channel': t['channel'], 'kernel': re.sub(
                r'\bu([ijk])\b', lambda m: source + m[1], t['kernel'])} for t in records]
            yield source, mapped

    for family, records in MULTIVARIATE_COUPLING_FAMILIES:
        for source, mapped in _map_family(records):
            yield family + ':' + source, local + mapped
    # Degree-3 contrast uses linear local F only; extra bilinear junk is a
    # derivative artifact. H-only families keep parent G.
    linear_local = _local_f_library(names)[1:1 + len(names)]
    for family, records in MULTIVARIATE_POLY3_FAMILIES:
        has_g = any(t['channel'] == 'G' for t in records)
        keep_g = [] if has_g else [t for t in terms if t['channel'] == 'G']
        for source, mapped in _map_family(records):
            yield family + ':' + source, linear_local + keep_g + mapped


def _refine_one(context, terms, target, node_mode='none', *, allow_contrast=True,
                coefficient_cutoff=1e-4, contribution_cutoff=1e-4, residual_allowance=1e-8,
                round_digits=4, max_denominator=32, simplify_coefficients=True):
    terms = normalize_vector_terms(context, terms)
    coef, metrics = vector_fit(context, terms, target, node_mode)
    cv = vector_crosscheck(context, terms, target, node_mode)
    n_couple = sum(1 for t in terms if t['channel'] in ('G', 'H'))
    trace = []
    parent_score = cv.get('max_heldout_rmse')
    best = None
    contrast_main = allow_contrast and node_mode == 'none' and (
        metrics['derivative_rmse'] > 1e-2 or n_couple > 3 or len(terms) > 6
    )
    contrast_poly3 = allow_contrast and node_mode == 'none' and metrics['derivative_rmse'] > 1e-4
    if contrast_main or contrast_poly3:
        for family_name, family in _family_trials(context, terms):
            if family_name.startswith('cubic_') and not contrast_poly3:
                continue
            if (not family_name.startswith('cubic_')) and not contrast_main:
                continue
            record = {'action': 'family_trial', 'family': family_name, 'component': target}
            try:
                trial = normalize_vector_terms(context, family)
                budget = 16 if family_name.startswith('cubic_') else 12
                if len(trial) > budget:
                    record['status'] = 'budget_skipped'
                    trace.append(record)
                    continue
                cc, mm = vector_fit(context, trial, target)
                check = vector_crosscheck(context, trial, target)
                score = check.get('max_heldout_rmse')
                eligible = (parent_score is not None and score is not None and
                            check['full_rank'] and mm['rank'] == mm['columns'] and
                            score < 0.5 * parent_score)
                record.update(status='compared', heldout_rmse=score, eligible=eligible)
                if eligible and (best is None or score < best[0]):
                    best = (score, family_name, trial, cc, mm, check)
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                record.update(status='domain_or_rank_skipped', reason=str(exc))
            trace.append(record)
        if best is not None:
            score, family_name, terms, coef, metrics, cv = best
            trace.append({'action': 'vector_family_contrast', 'family': family_name,
                          'component': target, 'heldout_rmse': score})
    limit = max(metrics['derivative_rmse'] * 1.2, metrics['derivative_rmse'] + residual_allowance)
    def acceptable(ts, cs, mode=node_mode):
        m = vector_metrics(context, ts, cs, target, mode)
        check = vector_crosscheck(context, ts, target, mode)
        cvlimit = max(cv.get('max_heldout_rmse', 0) * 1.2, cv.get('max_heldout_rmse', 0) + residual_allowance)
        return (m['derivative_rmse'] <= limit and m['rank'] == m['columns'] and
                (not check['available'] or (check['full_rank'] and check['max_heldout_rmse'] <= cvlimit)))
    # Resolve exact redundancy before scale-aware pruning, without adding columns.
    for _ in range(len(terms)):
        _, current = vector_fit(context, terms, target, node_mode)
        if current['rank'] == current['columns']:
            break
        removed = False
        for j in reversed(range(len(terms))):
            trial = terms[:j] + terms[j + 1:]
            cc, mm = vector_fit(context, trial, target, node_mode)
            check = vector_crosscheck(context, trial, target, node_mode)
            cvlimit = max(cv.get('max_heldout_rmse', 0) * 1.2,
                          cv.get('max_heldout_rmse', 0) + residual_allowance)
            if (mm['rank'] == current['rank'] and mm['derivative_rmse'] <= limit and
                    (not check['available'] or check['max_heldout_rmse'] <= cvlimit)):
                trace.append(dict(action='remove_redundant_column', component=target, term=terms[j]))
                terms, coef = trial, cc
                removed = True
                break
        if not removed:
            break
    for _ in range(len(terms)):
        A, y = vector_design(context, terms, target, node_mode)
        scale = max(float(np.sqrt(np.mean(y * y))), 1e-15)
        eligible = [
            j for j in range(len(terms))
            if abs(coef[j]) <= coefficient_cutoff and abs(coef[j]) * float(np.sqrt(np.mean(A[:, j] ** 2))) <= contribution_cutoff * scale
        ]
        removed = False
        for j in sorted(eligible, key=lambda k: abs(coef[k])):
            trial = terms[:j] + terms[j + 1:]
            cc, _mm = vector_fit(context, trial, target, node_mode)
            if acceptable(trial, cc):
                trace.append({'action': 'prune', 'component': target, 'term': terms[j]})
                terms, coef = trial, cc
                removed = True
                break
        if not removed:
            break
    for j in (range(len(terms)) if simplify_coefficients else []):
        candidates = [float(Fraction(float(coef[j])).limit_denominator(max_denominator))]
        candidates.extend(float(round(float(coef[j]), d)) for d in range(round_digits + 1))
        for value in dict.fromkeys(candidates):
            if abs(value - coef[j]) > max(1e-3, abs(coef[j]) * 1e-3):
                continue
            trial_c = np.asarray(coef, float).copy()
            trial_c[j] = value
            if node_mode != 'none':
                A, y = vector_design(context, terms, target, node_mode)
                k = len(terms)
                trial_c[k:] = solve(A[:, k:], y - A[:, :k] @ trial_c[:k])[0]
            if acceptable(terms, trial_c) or near_simple_round(coef[j], value):
                if value != coef[j]:
                    trace.append({'action': 'round', 'component': target, 'before': float(coef[j]), 'after': value})
                coef = trial_c
                break
    keep = [j for j in range(len(terms)) if coef[j] != 0]
    coef = np.concatenate([coef[keep], coef[len(terms):]])
    terms = [terms[j] for j in keep]
    metrics = vector_metrics(context, terms, coef, target, node_mode)
    return terms, coef, metrics, trace


def _node_attempt(context, terms, target, mode):
    key = '1' if mode == 'constant' else target + 'i'
    shared = [t for t in terms if not (t['channel'] == 'F' and t['kernel'].replace(' ', '') == key)]
    c, m = vector_fit(context, shared, target, mode)
    _, base = vector_fit(context, shared, target)
    cv = vector_crosscheck(context, shared, target, mode)
    baseline = vector_crosscheck(context, shared, target)
    node = c[len(shared):]
    spread = np.asarray(cv.get('coefficient_spread', [])[len(shared):])
    instability = float(spread.max()) if spread.size else float('inf')
    x = vector_fields(context)[target]
    lo, hi = np.quantile(x, [0.05, 0.95], axis=(0, 1))
    overlap = any(max(lo[i], lo[j]) < min(hi[i], hi[j]) and
                  abs(node[i] - node[j]) > max(10 * instability, 1e-10)
                  for i in range(len(node)) for j in range(i))
    checks = dict(multiple_trajectories=cv['available'], overlap=overlap,
                  identifiable=m['rank'] == m['columns'] and cv.get('full_rank', False),
                  stable=np.std(node) > max(10 * instability, 1e-10),
                  train_improves=m['derivative_rmse'] < .2 * max(base['derivative_rmse'], 1e-14),
                  heldout_improves=cv.get('max_heldout_rmse', float('inf')) <
                      .2 * max(baseline.get('max_heldout_rmse', 0), 1e-14))
    return shared, c, m, cv, {key: bool(value) for key, value in checks.items()}


def _keep_one_node_field(context, per, channels):
    """At most one node-indexed parameter on one variable, same rule as univariate."""
    node_names = [name for name, block in per.items() if block.get('node_mode', 'none') != 'none']
    if len(node_names) <= 1:
        return per, channels, []
    best = min(
        node_names,
        key=lambda name: float((channels.get(name) or {}).get('derivative_rmse', 1e9)),
    )
    dropped = []
    for name in node_names:
        if name == best:
            continue
        terms = per[name]['terms']
        coef, metrics = vector_fit(context, terms, name, 'none')
        per[name] = {
            'terms': terms,
            'coefficients': np.asarray(coef, float).tolist(),
            'node_mode': 'none',
        }
        if name in channels:
            channels[name]['node_mode'] = 'none'
            channels[name]['derivative_rmse'] = metrics['derivative_rmse']
        dropped.append(name)
    return per, channels, [{'action': 'keep_one_node_field', 'kept': best, 'dropped': dropped}]


def refine_multivariate(context, candidate_id, replacement_terms=None, evidence='', **options):
    old = get(context, candidate_id)
    if old.get('kind') not in ('multivariate', 'vector'):
        raise ValueError('requires vector candidate')
    options.update(clamp_refine_budget(
        coefficient_cutoff=options.get('coefficient_cutoff', 1e-4),
        contribution_cutoff=options.get('contribution_cutoff', 1e-4),
        residual_allowance=options.get('residual_allowance', 1e-8),
        round_digits=options.get('round_digits', 4),
        max_denominator=options.get('max_denominator', 32),
    ))
    replacements = {}
    if replacement_terms is not None:
        if not replacement_terms or len(replacement_terms) > 12 or len(evidence.strip()) < 12:
            raise ValueError('one <=12-term replacement family with residual evidence required')
        if any(b.get('node_mode', 'none') != 'none' for b in variable_blocks(old).values()):
            raise ValueError('replacement requires shared baseline; node authorization cannot transfer')
        for t in replacement_terms:
            target = t.get('component')
            if target not in old['components']:
                raise ValueError('each vector replacement record requires a valid component')
            replacements.setdefault(target, []).append({'channel': t['channel'], 'kernel': t['kernel']})
    per, traces, baseline, checks = {}, [], {}, {}
    rollback_per = {}
    for name, block in variable_blocks(old).items():
        terms = replacements.get(name, block['terms'])
        mode = block.get('node_mode', 'none')
        c, m = vector_fit(context, normalize_vector_terms(context, terms), name, mode)
        rollback_per[name] = dict(terms=normalize_vector_terms(context, terms), coefficients=c.tolist(), node_mode=mode)
        baseline[name] = {k: v for k, v in m.items() if k != 'residual'}
        terms, c, m, trace = _refine_one(context, terms, name, mode,
                                       allow_contrast=replacement_terms is None, **options)
        traces.extend(trace)
        per[name] = dict(terms=terms, coefficients=c.tolist(), node_mode=mode)
        cv = vector_crosscheck(context, terms, name, mode)
        unresolved = [t for j, t in enumerate(terms) if abs(c[j]) <= 1e-6]
        checks[name] = dict(rank_ok=m['rank'] == m['columns'], budget_ok=len(terms) <= 12,
                            unresolved_small_terms=unresolved, crosscheck=cv)
        if mode != 'none':
            checks[name]['node_guards'] = _node_attempt(context, terms, name, mode)[-1]
    short = all(v['rank_ok'] and v['budget_ok'] and not v['unresolved_small_terms'] and
                all(v.get('node_guards', {}).values()) for v in checks.values())
    short = short and sum(b['node_mode'] != 'none' for b in per.values()) <= 1
    rollback = multivariate_save(context, rollback_per, 'unrounded', {'parent_id': candidate_id})
    cid = multivariate_save(context, per, 'short', {'short_review': short, 'parent_id': candidate_id})
    return multivariate_summary(context, cid, 'hypergraph_reintegrate' if short else 'hypergraph_formula_refine; resolve short review or stop unresolved',
                          status='proposal', trace=traces, baseline_metrics=baseline,
                          short_review=short, review=checks, rollback_id=rollback)


def node_fit_multivariate(context, candidate_id, mode, evidence):
    old = get(context, candidate_id)
    if mode not in ('constant', 'linear') or len(evidence.strip()) < 12:
        raise ValueError('constant/linear and residual evidence required')
    if any(b.get('node_mode', 'none') != 'none' for b in variable_blocks(old).values()):
        raise ValueError('start with shared candidate; no second node-indexed parameter')
    trials, accepted = {}, []
    for name, block in variable_blocks(old).items():
        ts, c, m, cv, checks = _node_attempt(context, block['terms'], name, mode)
        trials[name] = checks
        if all(checks.values()):
            accepted.append((cv['max_heldout_rmse'], name, ts, c))
    if not accepted:
        return multivariate_summary(context, candidate_id, 'hypergraph_formula_refine or stop unresolved', status='rejected', checks=trials)
    _, name, ts, c = min(accepted, key=lambda item: item[0])
    per = copy.deepcopy(variable_blocks(old))
    per[name] = dict(terms=ts, coefficients=c.tolist(), node_mode=mode)
    cid = multivariate_save(context, per, 'node_fit', {'node_evidence': evidence, 'node_checks': trials[name]})
    return multivariate_summary(
        context, cid,
        'hypergraph_reintegrate on this candidate_id, or hypergraph_formula_refine first if coefficients still look unrounded',
        status='accepted_for_refinement', selected_component=name, selected_mode=mode, checks=trials,
    )


def vector_full_residual(context, candidate):
    fields, targets = vector_targets(context)
    ref = next(iter(fields.values()))
    residual = np.zeros(ref.shape + (len(candidate['components']),), float)
    for a, name in enumerate(candidate['components']):
        block = variable_blocks(candidate)[name]
        terms = block['terms']
        coef = np.asarray(block['coefficients'], float)
        mode = block.get('node_mode', 'none')
        A, y = vector_design(context, terms, name, mode)
        residual[..., a] = (y - A @ coef).reshape(ref.shape)
    return residual


def _rhs_fields(candidate, fields_1d, context):
    """fields_1d: name -> (N,) at one time."""
    names = candidate['components']
    n = next(iter(fields_1d.values())).shape[0]
    stacked = {name: arr.reshape(1, 1, n) for name, arr in fields_1d.items()}
    out = np.zeros((n, len(names)))
    for a, name in enumerate(names):
        block = variable_blocks(candidate)[name]
        terms = block['terms']
        coef = np.asarray(block['coefficients'], float)
        mode = block.get('node_mode', 'none')
        acc = np.zeros(n)
        for term, amp in zip(terms, coef[:len(terms)]):
            acc += amp * vector_column(context, term, stacked)[0, 0]
        if mode == 'constant':
            acc += coef[len(terms):]
        elif mode == 'linear':
            acc += coef[len(terms):] * fields_1d[name]
        out[:, a] = acc
    return out


def reintegrate_multivariate(context, candidate_id):
    cand = get(context, candidate_id)
    state = registry(context)
    state['gates'].pop(candidate_id, None)
    obs = host_state(context)['observations']
    if not isinstance(obs, dict) or 'x' not in obs or 't' not in obs:
        return {'status': 'blocked', 'can_submit': False,
                'next_action': 'host must supply full original vector x/t'}
    x = np.asarray(obs['x'], float)
    t = np.asarray(obs['t'], float)
    names = cand['components']
    d = len(names)
    if x.ndim != 4 or x.shape[-1] != d or x.shape[-2] != context.num_nodes:
        raise ValueError('vector observations must be (trajectory,time,node,component)')
    if t.ndim == 1:
        t = np.broadcast_to(t, (x.shape[0], len(t)))
    if (t.shape != x.shape[:2] or x.shape[1] < 2 or not np.isfinite(x).all() or
            not np.isfinite(t).all() or not (np.diff(t, axis=1) > 0).all() or
            x.shape[0] != len(next(iter(vector_fields(context).values())))):
        raise ValueError('invalid full vector grid or trajectory count')
    runs = []
    component_squared = np.zeros(d)
    squared = 0.0
    count = 0
    maximum = 0.0
    n = x.shape[2]
    try:
        for obs_x, grid in zip(x, t):
            calls = [0]
            def fun(_, vec):
                calls[0] += 1
                if calls[0] > 400000:
                    raise ValueError('integration evaluation budget exceeded')
                state_nd = vec.reshape(n, d)
                fields = {name: state_nd[:, a] for a, name in enumerate(names)}
                return _rhs_fields(cand, fields, context).ravel()
            y0 = obs_x[0].reshape(-1)
            sol = solve_ivp(
                fun, (float(grid[0]), float(grid[-1])), y0,
                t_eval=grid, method='DOP853', rtol=1e-9, atol=1e-11,
            )
            if not sol.success or not np.isfinite(sol.y).all():
                raise ValueError('incomplete/divergent integration')
            pred = sol.y.T.reshape(obs_x.shape)
            err = pred - obs_x
            mse = float(np.mean(err ** 2))
            mx = float(np.max(np.abs(err)))
            runs.append({'rmse': float(np.sqrt(mse)), 'max_abs': mx, 'nfev': sol.nfev})
            component_squared += np.sum(err ** 2, axis=(0, 1))
            squared += float(np.sum(err ** 2))
            count += err.size
            maximum = max(maximum, mx)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        result = {
            'status': 'failed', 'candidate_id': candidate_id, 'reason': str(exc),
            'can_submit': False, 'multivariate': True, 'vector': True,
            'next_action': reintegrate_next_action(candidate_id, passed=False),
        }
        state['gates'][candidate_id] = copy.deepcopy(result)
        return result
    rmse = float(np.sqrt(squared / count))
    passed = rmse <= 1e-5
    ready = bool(passed)
    result = {
        'status': 'passed' if passed else 'failed',
        'candidate_id': candidate_id,
        'resim_rmse': rmse,
        'per_variable_rmse': dict(zip(names, np.sqrt(component_squared / (count / d)).tolist())),
        'per_component_rmse': dict(zip(names, np.sqrt(component_squared / (count / d)).tolist())),
        'threshold': 1e-5,
        'max_abs': maximum,
        'per_trajectory': runs,
        'can_submit': ready,
        'multivariate': True,
        'vector': True,
        'data_signature': multivariate_signature(context),
        'symbolic_equivalence_proven': False,
        'next_action': reintegrate_next_action(candidate_id, passed=ready),
    }
    state['gates'][candidate_id] = copy.deepcopy(result)
    return json_safe(result)


def _kernel_to_symbols(text: str) -> str:
    text = re.sub(r'\b([A-Za-z]\w*)i\b', r'\1_i', text)
    text = re.sub(r'\b([A-Za-z]\w*)j\b', r'\1_j', text)
    text = re.sub(r'\b([A-Za-z]\w*)k\b', r'\1_k', text)
    return text


def _channel_tuple_expr(block, channel, name):
    terms = block['terms']
    coef = np.asarray(block['coefficients'], float)
    mode = block.get('node_mode', 'none')
    parts = []
    if channel == 'F' and mode == 'constant':
        parts.append(f'omega_{name}_i')
    elif channel == 'F' and mode == 'linear':
        parts.append(f'mu_{name}_i * {name}_i')
    for term, amp in zip(terms, coef[:len(terms)]):
        if term['channel'] != channel:
            continue
        parts.append(f'({amp:.12g}) * ({_kernel_to_symbols(term["kernel"])})')
    return ' + '.join(parts) if parts else '0'


def serialize_multivariate(context, candidate):
    names = candidate['components']
    exported = {}
    f_parts, g_parts, h_parts = [], [], []
    formula_parts = []
    for name in names:
        block = variable_blocks(candidate)[name]
        terms = block['terms']
        coef = np.asarray(block['coefficients'], float)
        mode = block.get('node_mode', 'none')
        if mode == 'constant':
            exported[f'omega_{name}'] = coef[len(terms):].tolist()
        elif mode == 'linear':
            exported[f'mu_{name}'] = coef[len(terms):].tolist()
        f_expr = _channel_tuple_expr(block, 'F', name)
        g_expr = _channel_tuple_expr(block, 'G', name)
        h_expr = _channel_tuple_expr(block, 'H', name)
        f_parts.append(f_expr)
        g_parts.append(g_expr)
        h_parts.append(h_expr)
        formula_parts.append(f'd{name}_dt = {f_expr} + sum({g_expr}) + sum({h_expr})')
    fgh = {
        'F': '(' + ', '.join(f_parts) + ')',
        'G': '(' + ', '.join(g_parts) + ')',
        'H': '(' + ', '.join(h_parts) + ')',
        'components': names,
    }
    if exported:
        host_state(context)['exported_node'] = {'name': 'multivariate', 'values': exported, 'mode': 'mixed'}
    host_state(context)['exported_vector'] = fgh
    return ' | '.join(formula_parts)


def multivariate_summary(context, cid, next_action, **extra):
    cand = get(context, cid)
    return json_safe({
        'candidate_id': cid,
        'multivariate': True,
        'vector': True,
        'components': {
            name: {
                'F': _format_channel(block['terms'], block['coefficients'][:len(block['terms'])], 'F'),
                'G': _format_channel(block['terms'], block['coefficients'][:len(block['terms'])], 'G'),
                'H': _format_channel(block['terms'], block['coefficients'][:len(block['terms'])], 'H'),
                'node_mode': block.get('node_mode', 'none'),
                **_component_report(context, name, block),
            }
            for name, block in variable_blocks(cand).items()
        },
        'can_submit': False,
        'next_action': next_action,
        **extra,
    })


def _component_report(context, name, block):
    terms, c = block['terms'], block['coefficients']
    m = vector_metrics(context, terms, c, name, block.get('node_mode', 'none'))
    residual = m.pop('residual')
    zero = [ch for ch in ('G', 'H') if not any(t['channel'] == ch and a != 0 for t, a in zip(terms, c))]
    suggestions = []
    for family, trial in _family_trials(context, terms):
        try:
            ts = normalize_vector_terms(context, trial)
            if len(ts) > 12:
                continue
            cv = vector_crosscheck(context, ts, name)
            if cv.get('full_rank'):
                suggestions.append((cv['max_heldout_rmse'], family))
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            continue
    return dict(**m, per_node_rmse=np.sqrt(np.mean(residual**2, axis=(0, 1))).tolist(),
                per_trajectory_rmse=np.sqrt(np.mean(residual**2, axis=(1, 2))).tolist(),
                zero_slots=zero, suggested_form_families=[f for _, f in sorted(suggestions)[:3]],
                suggestions_are_not_authorization=True)


VECTOR_FORM_KERNEL_FAMILIES = MULTIVARIATE_FORM_KERNEL_FAMILIES
VECTOR_COUPLING_FAMILIES = MULTIVARIATE_COUPLING_FAMILIES
VECTOR_POLY3_FAMILIES = MULTIVARIATE_POLY3_FAMILIES
diagnose_vector = diagnose_multivariate
screen_vector = screen_multivariate
refine_vector = refine_multivariate
node_fit_vector = node_fit_multivariate
reintegrate_vector = reintegrate_multivariate
serialize_vector = serialize_multivariate
vector_save = multivariate_save
vector_summary = multivariate_summary
vector_signature = multivariate_signature
