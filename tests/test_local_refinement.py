import pytest
import torch
from src.fwi.local_refinement import refine_gene,refine_population
from test_evaluator import make_evaluator


@pytest.mark.parametrize("optimizer", ["Adam","SGD","LBFGS"])
def test_local_refinement_preserves_or_improves_objective(optimizer):
    evaluator,env = make_evaluator(dtype=torch.float64)
    z = torch.zeros(16,dtype=torch.float64)
    initial = evaluator(z,env)
    best,result = refine_gene(z,initial,evaluator,env,3,optimizer,.1)
    assert result.total <= initial.total
    assert result.total < initial.total
    assert not best.requires_grad
    assert evaluator.solver.counters.model_evaluations >= 3


def test_topk_never_refines_all_individuals():
    evaluator,env = make_evaluator()
    genes = torch.zeros(3,16)
    results = evaluator.population(genes,env)
    _,_,_,count = refine_population(genes,results,evaluator,env,{"top_k":100})
    assert count == 2
