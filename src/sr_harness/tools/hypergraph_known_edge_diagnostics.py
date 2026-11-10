"""Audit empty channels, weights, counting, and residual clues. Does not search edges."""
import numpy as np
from .base_tool import BaseTool, ToolMetadata
from ._hypergraph_known_edge_common import FORM_KERNEL_FAMILIES, inputs, is_multivariate, kernel, relations, get, rhs, readable


def _finite_corr(left, right) -> float:
    a = np.asarray(left, float).ravel()
    b = np.asarray(right, float).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 32:
        return 0.0
    a, b = a[mask], b[mask]
    if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def form_residual_clues(x, residual, rel) -> dict:
    """Rank short form families by residual correlation. Clues are not exclusive."""
    signed = bool(x.min() < -0.2 and x.max() > 0.2)
    nonnegative = bool(x.min() >= -1e-6)
    stats = {'signed_domain': signed, 'nonnegative_domain': nonnegative, 'family_scores': {}}
    ranked = []
    for name, family in FORM_KERNEL_FAMILIES:
        corrs = []
        for term in family:
            rows = rel[term['channel']][1]
            if not len(rows):
                continue
            vals = {'xi': x[..., rows[:, 0]], 'xj': x[..., rows[:, 1]]}
            if term['channel'] == 'H':
                vals['xk'] = x[..., rows[:, 2]]
            try:
                probe = kernel(term['kernel'], vals)
            except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                continue
            if not np.isfinite(probe).all():
                continue
            corrs.append(abs(_finite_corr(residual[..., rows[:, 0]], probe)))
        if not corrs:
            continue
        score = float(np.mean(corrs))
        stats['family_scores'][name] = score
        if score > 0.2:
            ranked.append((score, name))
    ranked.sort(reverse=True)
    stats['suggested_form_families'] = [name for _, name in ranked]
    stats['form_reasons'] = [
        f'{name} residual correlation {score:.3f}' for score, name in ranked[:6]
    ]
    return stats

@BaseTool.register('hypergraph_known_edge_diagnostics')
class HypergraphKnownEdgeDiagnosticsTool(BaseTool):
    metadata=ToolMetadata(name='hypergraph_known_edge_diagnostics',description='Audit installed univariate or multivariate node variables, counting, weights and residual clues. Never searches edges.')

    def execute(self, candidate_id: str = '') -> dict:
        """Inspect context before screen or residuals after a candidate.

        Args:
            candidate_id: Optional ID returned by screen/refine/node tool.
        """
        if is_multivariate(self.context):
            from ._hypergraph_multivariate import diagnose_multivariate
            return diagnose_multivariate(self.context, candidate_id)
        try: x,y=inputs(self.context); rel=relations(self.context)
        except (ValueError,KeyError) as e:
            return {'status':'blocked','reason':str(e),'can_submit':False,'next_action':'fix host univariate x/dx_dt contract; multivariate named variables are not blocked'}
        incoming=np.zeros(x.shape[-1],int)
        channels={}
        for ch,(name,rows,w) in rel.items():
            np.add.at(incoming,rows[:,0],1)
            channels[ch]={'relation':name,'rows':len(rows),'empty_convention':not len(rows),
                          'weights_supplied':name+'_weights' in self.context.data,
                          'effective_weight_range':[float(w.min()),float(w.max())] if len(w) else None}
        residual=y if not candidate_id else y-rhs(self.context,get(self.context,candidate_id),x)
        residual_rmse=float(np.sqrt(np.mean(residual**2)))
        clues={}
        for mode in ('constant','linear'):
            z=np.ones_like(x) if mode=='constant' else x
            denom=np.sum(z*z,axis=1)
            safe=denom>1e-12*np.maximum(np.max(denom),1)
            estimates=np.divide(np.sum(z*residual,axis=1),denom,out=np.zeros_like(denom),where=safe)
            unexplained=residual-estimates[:,None,:]*z
            clues[mode]={'all_nodes_excited':bool(safe.all()),
                         'cross_trajectory_max_spread':float(np.max(np.ptp(estimates,axis=0))),
                         'between_node_std':float(np.std(np.mean(estimates,axis=0))),
                         'remaining_rmse':float(np.sqrt(np.mean(unexplained**2)))}
        rc=clues['constant']['remaining_rmse']; rl=clues['linear']['remaining_rmse']
        recommended='none'
        if min(rc, rl) <= 0.7 * max(residual_rmse, 1e-15):
            if rl < 0.85 * rc:
                recommended='linear'
            elif rc < 0.85 * rl:
                recommended='constant'
            else:
                recommended='linear' if rl <= rc else 'constant'
        n_couple=0
        if candidate_id:
            n_couple=sum(1 for t in get(self.context,candidate_id)['terms'] if t['channel'] in ('G','H'))
        form_clues = form_residual_clues(x, residual, rel) if candidate_id else {}
        if not candidate_id:
            next_action='hypergraph_formula_screen'
        elif recommended != 'none':
            next_action=(
                f'hypergraph_node_parameter_fit(mode={recommended!r}); if the current shared support is a long soup, '
                'the tool contrasts short product / contagion / phase-difference families. '
                'Do not add x[i]**2 or extra shared kernels to fake node rates.'
            )
        else:
            families = form_clues.get('suggested_form_families') or []
            if residual_rmse > 1e-2 or n_couple > 3:
                hint = ', '.join(families[:4]) if families else 'the short-form catalog'
                next_action = (
                    f'hypergraph_formula_refine; it contrasts admissible short families ({hint}). '
                    'Pass replacement_terms for one named family if you have a stronger residual reason. '
                    'Do not dump several families into one long formula.'
                )
            else:
                next_action='hypergraph_formula_refine; do not raise local polynomial order to absorb leftover residual'
        return {'status':'ok','shape':list(x.shape),'state_range':[float(x.min()),float(x.max())],
                'derivative_rms':float(np.sqrt(np.mean(y*y))), 'channels':channels,
                'isolated_target_nodes':np.flatnonzero(incoming==0).tolist(),
                'residual_rmse':residual_rmse,
                'node_clues':clues,'recommended_node_mode':recommended,
                'form_clues':form_clues,
                'shared_coupling_term_count':n_couple,
                'clues_are_not_authorization':True,'can_submit':False,
                'full_observations_available':'_known_edges_observations' in self.context.data,
                'next_action':next_action}

    @classmethod
    def format_result_dict(cls,result): return readable(result)
