"""Guarded node constants or slopes, fitted together with the shared coupling."""
import numpy as np
from .base_tool import BaseTool, ToolMetadata
from ._hypergraph_known_edge_common import (
    get, inputs, fit, crosscheck, save, summary, readable, normalize_terms,
)

# Scientific short coupling families for node-heterogeneous F, not a per-case kernel list.
# Product / gated contagion / phase-difference sine cover the usual shared G,H while
# F_i is a node constant or node slope. Form-library leftovers belong in screen, not here.
SHORT_COUPLING_FAMILIES = (
    (('G', 'xi * xj'), ('H', 'xi * xj * xk')),
    (('G', '(1 - xi) * xj'), ('H', '(1 - xi) * xj * xk')),
    (('G', 'sin(xj - xi)'), ('H', 'sin(xj + xk - 2 * xi)')),
)


def _family_terms(records):
    return [{'channel': channel, 'kernel': kernel} for channel, kernel in records]


@BaseTool.register('hypergraph_node_parameter_fit')
class HypergraphNodeParameterFitTool(BaseTool):
    metadata=ToolMetadata(name='hypergraph_node_parameter_fit',description='Test one node-indexed constant or one slope when the residual is node-wise. Requires cross-trajectory improvement and identifiability.')

    def _attempt(self, terms, mode):
        key='1' if mode=='constant' else 'xi'
        shared=[t for t in terms if not (t['channel']=='F' and t['kernel'].replace(' ','')==key)]
        _,base=fit(self.context,shared)
        base_cv=crosscheck(self.context,shared)
        c,m=fit(self.context,shared,mode); cv=crosscheck(self.context,shared,mode)
        node=c[len(shared):]; spread=np.asarray(cv['coefficient_spread'][len(shared):])
        heterogeneity=float(np.std(node)); instability=float(np.max(spread)) if spread.size else 0.0
        x,_=inputs(self.context)
        lo=np.quantile(x,0.05,axis=(0,1)); hi=np.quantile(x,0.95,axis=(0,1))
        overlap_pairs=sum(bool(max(lo[i],lo[j])<min(hi[i],hi[j]) and
                               abs(node[i]-node[j])>max(10*instability,1e-10))
                          for i in range(len(node)) for j in range(i))
        checks={'overlapping_state_evidence':overlap_pairs>0,
                'shared_error_reduced':m['derivative_rmse'] < 0.2*max(base['derivative_rmse'],1e-14),
                'heldout_improves':cv['max_heldout_rmse'] < 0.2*max(base_cv.get('max_heldout_rmse',0),1e-14) if cv.get('available') else False,
                'identifiable':m['rank']==m['columns'] and bool(cv.get('full_rank')),
                'heterogeneity_repeats':heterogeneity>max(10*instability,1e-10)}
        return shared,c,m,cv,base,base_cv,checks

    def execute(self, candidate_id: str, mode: str, evidence: str) -> dict:
        """Propose F_i only if measured stability and held-out improvement support it.

        Args:
            candidate_id: Shared candidate ID from screen/refine.
            mode: constant or linear; never per-node nonlinear couplings.
            evidence: Observed isolated-node or repeated residual clue, not a case ID.
        """
        old=get(self.context,candidate_id)
        if old.get('kind') in ('multivariate', 'vector'):
            from ._hypergraph_multivariate import node_fit_multivariate
            return node_fit_multivariate(self.context, candidate_id, mode, evidence)
        if mode not in ('constant','linear') or len(evidence.strip())<12:
            raise ValueError('supply constant/linear mode and an observed residual explanation')
        if old['node_mode']!='none': raise ValueError('start from shared baseline; no stacking node freedoms')
        if len(inputs(self.context)[0])<2:
            return {'status':'blocked','can_submit':False,'next_action':'need multiple trajectories for node-parameter evidence'}

        parent_error=None
        p_checks=None
        n_gh=sum(1 for t in old['terms'] if t['channel'] in ('G','H'))
        n_f=sum(1 for t in old['terms'] if t['channel']=='F')
        parent_is_short = n_gh <= 2 and n_f <= 1
        try:
            _, parent_shared = fit(self.context, old['terms'])
            parent_rmse=float(parent_shared['derivative_rmse'])
        except (ValueError, np.linalg.LinAlgError) as exc:
            parent_rmse=float('inf'); parent_error=str(exc)
        try:
            p_terms,p_c,p_m,p_cv,p_base,p_base_cv,p_checks=self._attempt(old['terms'], mode)
        except (ValueError, np.linalg.LinAlgError) as exc:
            parent_error=str(exc)
        else:
            # A long shared soup can pass a 5× residual drop without being the
            # coupling family; those candidates fail reintegration. Only keep a
            # short parent; otherwise contrast the scientific coupling families.
            if parent_is_short and all(p_checks.values()):
                cid=save(self.context,p_terms,p_c,mode,'node_fit',{'node_evidence':evidence,'node_checks':p_checks})
                return summary(self.context,cid,'hypergraph_formula_refine (round shared coefficients, refit node parameters)',
                               status='accepted_for_refinement',checks=p_checks,shared_metrics=p_base,
                               crosscheck=p_cv,coupling_source='parent')
        if parent_is_short or parent_rmse <= 1e-2:
            return {'status':'rejected','checks':p_checks,'parent_error':parent_error,
                    'parent_shared_rmse':None if parent_rmse==float('inf') else parent_rmse,
                    'can_submit':False,
                    'next_action':'hypergraph_formula_refine: compare a short shared family or resolve rank/gauge; do not add higher node powers'}

        # Parent soup often absorbs F_i into extra shared kernels; those fits fail
        # identifiability. Contrast a few short shared G/H families, then the same guards.
        accepted=[]
        contrast=[]
        for family in SHORT_COUPLING_FAMILIES:
            try:
                terms=normalize_terms(self.context, _family_terms(family))
            except (ValueError, FloatingPointError, OverflowError) as exc:
                contrast.append({'family':family,'status':'invalid','reason':str(exc)})
                continue
            if not terms:
                contrast.append({'family':family,'status':'empty_after_channel_drop'})
                continue
            for hyp in ('constant','linear'):
                try:
                    ts,c,m,cv,base,base_cv,checks=self._attempt(terms, hyp)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    contrast.append({'family':family,'mode':hyp,'status':'error','reason':str(exc)})
                    continue
                item={'family':[t['kernel'] for t in ts],'mode':hyp,'checks':checks,
                      'derivative_rmse':m['derivative_rmse'],
                      'max_heldout_rmse':cv.get('max_heldout_rmse')}
                contrast.append(item)
                if all(checks.values()):
                    accepted.append((float(cv.get('max_heldout_rmse', m['derivative_rmse'])),
                                     float(m['derivative_rmse']), hyp, ts, c, m, cv, base, checks))
        if accepted:
            accepted.sort(key=lambda row: (row[0], row[1], 0 if row[2]==mode else 1))
            _,_,hyp,ts,c,m,cv,base,checks=accepted[0]
            cid=save(self.context,ts,c,hyp,'node_fit',{
                'node_evidence':evidence,'node_checks':checks,
                'coupling_source':'short_family_contrast','requested_mode':mode,
            })
            return summary(self.context,cid,'hypergraph_formula_refine (round shared coefficients, refit node parameters)',
                           status='accepted_for_refinement',checks=checks,shared_metrics=base,
                           crosscheck=cv,coupling_source='short_family_contrast',selected_mode=hyp,
                           parent_checks=p_checks,parent_error=parent_error,contrast=contrast)
        return {'status':'rejected','checks':p_checks,'parent_error':parent_error,'contrast':contrast,
                'shared_metrics':None,'can_submit':False,
                'next_action':'hypergraph_formula_refine: compare a short shared family or resolve rank/gauge; do not add higher node powers'}

    @classmethod
    def format_result_dict(cls,result): return readable(result)
