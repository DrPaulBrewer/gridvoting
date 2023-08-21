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

def test_grid_spatial_utility():
    # this also tests gridvoting github issue #10
    import gridvoting as gv
    xp = gv.xp
    grid = gv.Grid(x0=-5,x1=5,y0=-7,y1=7)
    assert grid.gshape == (15,11)
    u = grid.spatial_utilities(voter_ideal_points=[[0,1]]).reshape(grid.gshape)
# this reshaped output is correct, the dependence on the squared distance from the ideal point is clear
    correct_u = xp.array([
           [-61., -52., -45., -40., -37., -36., -37., -40., -45., -52., -61.],
           [-50., -41., -34., -29., -26., -25., -26., -29., -34., -41., -50.],
           [-41., -32., -25., -20., -17., -16., -17., -20., -25., -32., -41.],
           [-34., -25., -18., -13., -10.,  -9., -10., -13., -18., -25., -34.],
           [-29., -20., -13.,  -8.,  -5.,  -4.,  -5.,  -8., -13., -20., -29.],
           [-26., -17., -10.,  -5.,  -2.,  -1.,  -2.,  -5., -10., -17., -26.],
           [-25., -16.,  -9.,  -4.,  -1.,  -0.,  -1.,  -4.,  -9., -16., -25.],
           [-26., -17., -10.,  -5.,  -2.,  -1.,  -2.,  -5., -10., -17., -26.],
           [-29., -20., -13.,  -8.,  -5.,  -4.,  -5.,  -8., -13., -20., -29.],
           [-34., -25., -18., -13., -10.,  -9., -10., -13., -18., -25., -34.],
           [-41., -32., -25., -20., -17., -16., -17., -20., -25., -32., -41.],
           [-50., -41., -34., -29., -26., -25., -26., -29., -34., -41., -50.],
           [-61., -52., -45., -40., -37., -36., -37., -40., -45., -52., -61.],
           [-74., -65., -58., -53., -50., -49., -50., -53., -58., -65., -74.],
           [-89., -80., -73., -68., -65., -64., -65., -68., -73., -80., -89.]])
    xp.testing.assert_array_equal(u,correct_u)
    
