"""Free integration of every original trajectory from its observed initial state."""
import numpy as np
from scipy.integrate import solve_ivp
from .base_tool import BaseTool, ToolMetadata
from ._hypergraph_known_edge_common import get, host_state, rhs, registry, signature, readable, reintegrate_next_action

@BaseTool.register('hypergraph_reintegrate')
class HypergraphReintegrateTool(BaseTool):
    metadata=ToolMetadata(name='hypergraph_reintegrate',description='Integrate every original trajectory from its observed initial state. Whole-trajectory RMSE<=1e-5 sets can_submit. This tool does not call submit_formula.')

    def execute(self, candidate_id: str) -> dict:
        """Run strict full-grid self-check; never fit parameters or initial states.

        Args:
            candidate_id: Candidate from refine, node-indexed fit, or a rollback ID.
        """
        c=get(self.context,candidate_id)
        if c.get('kind') in ('multivariate', 'vector'):
            from ._hypergraph_multivariate import reintegrate_multivariate
            return reintegrate_multivariate(self.context, candidate_id)
        state=registry(self.context); state['gates'].pop(candidate_id,None)
        obs=host_state(self.context)['observations'] or self.context.data.get('_known_edges_observations')
        if not isinstance(obs,dict) or 'x' not in obs or 't' not in obs:
            return {'status':'blocked','can_submit':False,'next_action':'host must supply full original x/t; do not substitute cropped derivative grid'}
        x,t=np.asarray(obs['x'],float),np.asarray(obs['t'],float)
        if x.ndim!=3 or x.shape[-1]!=self.context.num_nodes or not np.isfinite(x).all(): raise ValueError('invalid full scalar observations')
        if t.ndim==1: t=np.broadcast_to(t,(x.shape[0],len(t)))
        if t.shape!=x.shape[:2] or x.shape[1]<2 or not np.isfinite(t).all() or not (np.diff(t,axis=1)>0).all(): raise ValueError('invalid original time grid')
        if x.shape[0]!=np.asarray(self.context.data['x']).shape[0]: raise ValueError('full and fit trajectory counts mismatch')
        runs=[]; squared=0.; count=0; maximum=0.
        try:
            for obs_x,grid in zip(x,t):
                calls=[0]
                def fun(_,v):
                    calls[0]+=1
                    if calls[0]>200000: raise ValueError('integration evaluation budget exceeded')
                    return rhs(self.context,c,v)
                sol=solve_ivp(fun,(float(grid[0]),float(grid[-1])),obs_x[0].copy(),
                              t_eval=grid,method='DOP853',rtol=1e-11,atol=1e-13)
                if not sol.success or sol.y.shape!=obs_x.T.shape or not np.isfinite(sol.y).all(): raise ValueError('incomplete/divergent integration')
                err=sol.y.T-obs_x; mse=float(np.mean(err**2)); mx=float(np.max(np.abs(err)))
                runs.append({'rmse':float(np.sqrt(mse)),'max_abs':mx,'nfev':sol.nfev})
                squared+=float(np.sum(err**2)); count+=err.size; maximum=max(maximum,mx)
        except (ValueError,FloatingPointError,OverflowError) as e:
            result={'status':'failed','candidate_id':candidate_id,'reason':str(e),'can_submit':False,
                    'next_action':reintegrate_next_action(candidate_id, passed=False)}
            state['gates'][candidate_id]=result.copy()
            return result
        rmse=float(np.sqrt(squared/count)); passed=rmse<=1e-5
        ready=bool(passed)
        result={'status':'passed' if passed else 'failed','candidate_id':candidate_id,
                'resim_rmse':rmse,'threshold':1e-5,'max_abs':maximum,'per_trajectory':runs,
                'can_submit':ready,'data_signature':signature(self.context),
                'symbolic_equivalence_proven':False,'scope':'same-data full-trajectory consistency, not independent extrapolation',
                'next_action':reintegrate_next_action(candidate_id, passed=ready)}
        state['gates'][candidate_id]=result.copy()
        return result

    @classmethod
    def format_result_dict(cls,result): return readable(result)
