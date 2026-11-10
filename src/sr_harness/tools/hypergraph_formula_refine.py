"""Small family contrast, refit, scale-aware pruning and coefficient simplification.
No new BIC/Lasso pipeline and no case-specific family or parameter tables.
"""
from fractions import Fraction
import numpy as np
from .base_tool import BaseTool, ToolMetadata
from ._hypergraph_known_edge_common import (FORM_KERNEL_FAMILIES, get, normalize_terms, design, solve,
                                  fit, crosscheck, metrics, save, summary, readable,
                                  clamp_refine_budget, near_simple_round,
                                  consensus_shared_coefficients, subset_exact_ok)

@BaseTool.register('hypergraph_formula_refine')
class HypergraphFormulaRefineTool(BaseTool):
    metadata=ToolMetadata(name='hypergraph_formula_refine',description='Refit a screen support or one evidence-based short family; prune tiny contributions and try simple coefficients. Reintegration remains mandatory.')

    def execute(self, candidate_id: str, replacement_terms: list[dict] = None,
                evidence: str = '', coefficient_cutoff: float = 1e-4,
                contribution_cutoff: float = 1e-4, residual_allowance: float = 1e-8,
                round_digits: int = 4, max_denominator: int = 32,
                simplify_coefficients: bool = True) -> dict:
        """Generate a short candidate and an explicit rollback ID.

        Args:
            candidate_id: ID from screen, node_fit or previous refinement.
            replacement_terms: Optional <=12 channel/kernel records for ONE alternate family.
                Multivariate records require a variable name in component; replace that variable in full, keep unlisted variables.
            evidence: Residual/domain reason for replacing the family.
            coefficient_cutoff: Absolute coefficient threshold for deletion proposals.
            contribution_cutoff: Relative RMS contribution threshold; both cutoffs must hold.
            residual_allowance: Absolute derivative RMSE slack for proposals; NOT a submit threshold.
            round_digits: Coarsest allowed decimal search extends through this many digits (0-10).
            max_denominator: Try small rational coefficients up to this denominator (1-100).
            simplify_coefficients: False preserves fitted coefficients for rollback testing.
        """
        old=get(self.context,candidate_id)
        budget = clamp_refine_budget(
            coefficient_cutoff=coefficient_cutoff, contribution_cutoff=contribution_cutoff,
            residual_allowance=residual_allowance, round_digits=round_digits,
            max_denominator=max_denominator,
        )
        coefficient_cutoff = budget['coefficient_cutoff']
        contribution_cutoff = budget['contribution_cutoff']
        residual_allowance = budget['residual_allowance']
        round_digits = budget['round_digits']
        max_denominator = budget['max_denominator']
        if old.get('kind') in ('multivariate', 'vector'):
            from ._hypergraph_multivariate import refine_multivariate
            return refine_multivariate(
                self.context, candidate_id, replacement_terms, evidence,
                coefficient_cutoff=coefficient_cutoff, contribution_cutoff=contribution_cutoff,
                residual_allowance=residual_allowance, round_digits=round_digits,
                max_denominator=max_denominator, simplify_coefficients=simplify_coefficients,
            )
        mode=old['node_mode']
        if replacement_terms is not None:
            if len(evidence.strip())<12 or len(replacement_terms)>12: raise ValueError('one <=12-term family with observation evidence required')
            if mode!='none': raise ValueError('compare replacement families on shared baseline first; node authorization does not transfer')
        terms=normalize_terms(self.context,replacement_terms if replacement_terms is not None else old['terms'])
        c,base=fit(self.context,terms,mode); cvbase=crosscheck(self.context,terms,mode)
        rollback=save(self.context,terms,c,mode,'unrounded',{'parent_id':candidate_id})
        trace=[]
        # If the shared residual is still large, or the parent is a long soup,
        # contrast the short-form catalog. Replace only when held-out error is
        # clearly better (half), so a slightly luckier extra kernel is ignored.
        n_couple = sum(1 for t in old['terms'] if t['channel'] in ('G', 'H'))
        if replacement_terms is None and mode == 'none' and (
                base['derivative_rmse'] > 1e-2 or n_couple > 3):
            best_score = float(cvbase.get('max_heldout_rmse', base['derivative_rmse']) or base['derivative_rmse'])
            best = None
            local_f = [
                t for t in old['terms']
                if t['channel'] == 'F' and t['kernel'].replace(' ', '') in {'xi', 'xi*(1-xi)'}
            ]
            parent_long = n_couple > 3
            for family_name, family in FORM_KERNEL_FAMILIES:
                try:
                    ts = normalize_terms(self.context, local_f + list(family))
                    if not ts:
                        continue
                    cc, mm = fit(self.context, ts, mode)
                    cv = crosscheck(self.context, ts, mode)
                    score = float(cv.get('max_heldout_rmse', mm['derivative_rmse']) or mm['derivative_rmse'])
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                    continue
                clearly_better = score < 0.5 * best_score
                parent_held = float(
                    cvbase.get('max_heldout_rmse', base['derivative_rmse']) or base['derivative_rmse']
                )
                short_ok = parent_long and score <= 2.0 * parent_held
                if clearly_better or short_ok:
                    if best is None or score < best_score:
                        best_score = score
                        best = (family_name, ts, cc, mm, cv)
            if best is not None:
                family_name, terms, c, base, cvbase = best
                trace.append({'action': 'form_family_contrast', 'family': family_name,
                              'heldout_rmse': best_score})
        limit=max(base['derivative_rmse']*1.2,base['derivative_rmse']+residual_allowance)
        cvlimit=max(cvbase.get('max_heldout_rmse',0)*1.2,cvbase.get('max_heldout_rmse',0)+residual_allowance)
        def acceptable(ts,cs):
            m=metrics(self.context,ts,cs,mode)
            # Fixed-candidate trajectory errors accompany genuine refit holdouts.
            check=crosscheck(self.context,ts,mode)
            return (m['derivative_rmse']<=limit and m['rank']==m['columns'] and
                    (not check['available'] or (check['full_rank'] and check['max_heldout_rmse']<=cvlimit)))
        # Resolve exact design redundancies before threshold deletion; preserves predictions.
        changed=True
        while changed:
            changed=False; A,y=design(self.context,terms,mode)
            _,rank,_=solve(A,y)
            if rank==A.shape[1]: break
            for j in reversed(range(len(terms))):
                trial=terms[:j]+terms[j+1:]; cc,mm=fit(self.context,trial,mode)
                if mm['rank']==rank and mm['derivative_rmse']<=limit:
                    trace.append({'action':'remove_redundant_column','term':terms[j]})
                    terms,c=trial,cc; changed=True; break
            if not changed: break
        # Sequential refit avoids deleting several individually-small terms at once.
        for _ in range(len(terms)):
            A,y=design(self.context,terms,mode); scale=max(float(np.sqrt(np.mean(y*y))),1e-15)
            eligible=[j for j in range(len(terms)) if abs(c[j])<=coefficient_cutoff and abs(c[j])*float(np.sqrt(np.mean(A[:,j]**2)))<=contribution_cutoff*scale]
            removed=False
            for j in sorted(eligible,key=lambda j:abs(c[j])):
                trial=terms[:j]+terms[j+1:]; cc,_=fit(self.context,trial,mode)
                if acceptable(trial,cc):
                    trace.append({'action':'prune','term':terms[j]}); terms,c=trial,cc; removed=True; break
            if not removed: break
        unrounded=save(self.context,terms,c,mode,'unrounded',{'parent_id':rollback})
        if mode == 'none':
            locked = consensus_shared_coefficients(self.context, terms, c)
            if locked is not None:
                cc = np.asarray(locked, float)
                trace.append({'action': 'lock_clean_nodes', 'before': [float(v) for v in c],
                              'after': [float(v) for v in cc]})
                c = cc
        # Try coarse values first; never round the node vector to a canned template.
        for j in (range(len(terms)) if simplify_coefficients else []):
            candidates=[float(round(float(c[j]),d)) for d in range(round_digits+1)]
            candidates += [float(Fraction(float(c[j])).limit_denominator(max_denominator))]
            for v in dict.fromkeys(candidates):
                delta = abs(v-c[j])
                local_only = delta > max(1e-3, abs(c[j])*1e-3)
                cc=c.copy(); cc[j]=v
                if mode!='none':
                    A,y=design(self.context,terms,mode); k=len(terms)
                    cc[k:]=solve(A[:,k:],y-A[:,:k]@cc[:k])[0]
                if local_only and not subset_exact_ok(self.context, terms, cc, mode):
                    continue
                if acceptable(terms,cc) or near_simple_round(c[j], v) or subset_exact_ok(self.context, terms, cc, mode):
                    if v!=c[j]: trace.append({'action':'round','term':terms[j],'before':float(c[j]),'after':v})
                    c=cc; break
        # Remove zero amplitudes after rounding so output is actually short.
        keep=[j for j in range(len(terms)) if c[j]!=0]
        c=np.concatenate([c[keep],c[len(terms):]]); terms=[terms[j] for j in keep]
        m=metrics(self.context,terms,c,mode)
        A,y=design(self.context,terms,mode)
        scale=max(float(np.sqrt(np.mean(y*y))),1e-15)
        unresolved=[t for j,t in enumerate(terms) if abs(c[j])<=1e-4 and
                    abs(c[j])*float(np.sqrt(np.mean(A[:,j]**2)))<=1e-4*scale]
        final_cv=crosscheck(self.context,terms,mode)
        node_stable=True
        if mode!='none':
            spread=np.asarray(final_cv.get('coefficient_spread',[])[len(terms):])
            node_stable=bool(final_cv['available'] and final_cv['full_rank'] and spread.size and
                             np.std(c[len(terms):])>max(10*float(spread.max()),1e-10))
        short_ok=len(terms)<=12 and m['rank']==m['columns'] and not unresolved and node_stable
        cid=save(self.context,terms,c,mode,'short',{'short_review':short_ok,'parent_id':candidate_id})
        return summary(self.context,cid,'hypergraph_reintegrate' if short_ok else 'resolve redundant terms; do not submit',
                       status='proposal',trace=trace,baseline_metrics=base,crosscheck=crosscheck(self.context,terms,mode),
                       rollback_id=unrounded,original_refit_id=rollback,
                       short_review=short_ok,unresolved_small_terms=unresolved,warning='Derivative acceptance is provisional. Reintegration decides; no automatic symbolic truth claim.')

    @classmethod
    def format_result_dict(cls,result): return readable(result)
