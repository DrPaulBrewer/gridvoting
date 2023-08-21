import pytest

def test_module():
    import gridvoting
    assert (not gridvoting.use_cupy) == (gridvoting.xp is gridvoting.np)
    print("use_cupy is ",gridvoting.use_cupy)

def test_condorcet_zi():
    import gridvoting as gv
    xp = gv.xp
    condorcet_model_with_zi =  gv.CondorcetCycle(zi=True)
    assert not condorcet_model_with_zi.analyzed
    condorcet_model_with_zi.analyze()
    assert condorcet_model_with_zi.analyzed
    gv.xp.testing.assert_array_almost_equal(
        condorcet_model_with_zi.stationary_distribution,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )
    mc=condorcet_model_with_zi.MarkovChain
    alt = mc.solve_for_unit_eigenvector()
    gv.xp.testing.assert_array_almost_equal(
        alt,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )


