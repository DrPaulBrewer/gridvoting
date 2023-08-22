import pytest

def test_module():
    import gridvoting
    assert (not gridvoting.use_cupy) == (gridvoting.xp is gridvoting.np)
    print("use_cupy is ",gridvoting.use_cupy)

def test_grid_init():
    import gridvoting
    np = gridvoting.np
    grid = gridvoting.Grid(x0=-5,x1=5,y0=-7,y1=7)
    assert grid.x0 == -5
    assert grid.x1 == 5
    assert grid.xstep == 1
    assert grid.y0 == -7
    assert grid.y1 == 7
    assert grid.gshape == (15,11)
    assert grid.extent == (-5,5,-7,7)
    assert grid.len == 165
    correct_grid_x = np.array([
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5],
       [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]
       ])
    correct_grid_y = np.array(
       [
       [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
       [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
       [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
       [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
       [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
       [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
       [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
       [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
       [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
       [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6],
       [-7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7]
       ]
    )
    assert grid.x.shape == (165,)
    assert grid.y.shape == (165,)
    np.testing.assert_array_equal(grid.x.reshape(grid.gshape), correct_grid_x)
    np.testing.assert_array_equal(grid.y.reshape(grid.gshape), correct_grid_y)
 

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
    

def test_condorcet_zi():
    import gridvoting as gv
    xp = gv.xp
    condorcet_model_with_zi =  gv.CondorcetCycle(zi=True)
    assert not condorcet_model_with_zi.analyzed
    condorcet_model_with_zi.analyze()
    assert condorcet_model_with_zi.analyzed
    mc = condorcet_model_with_zi.MarkovChain
    gv.xp.testing.assert_array_almost_equal(
        mc.P,
        xp.array([
            [ 2/3, 0, 1/3],
            [ 1/3, 2/3, 0],
            [ 0,  1/3, 2/3]
        ]),
        decimal=10
    )
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

def test_condorcet_mi():
    import gridvoting as gv
    xp = gv.xp
    condorcet_model_with_mi =  gv.CondorcetCycle(zi=False)
    assert not condorcet_model_with_mi.analyzed
    condorcet_model_with_mi.analyze()
    assert condorcet_model_with_mi.analyzed
    mc = condorcet_model_with_mi.MarkovChain
    gv.xp.testing.assert_array_almost_equal(
        mc.P,
        xp.array([
            [ 1/2, 0, 1/2],
            [ 1/2, 1/2, 0],
            [ 0,  1/2, 1/2]
        ]),
        decimal=10
    )
    gv.xp.testing.assert_array_almost_equal(
        condorcet_model_with_mi.stationary_distribution,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )
    mc=condorcet_model_with_mi.MarkovChain
    alt = mc.solve_for_unit_eigenvector()
    gv.xp.testing.assert_array_almost_equal(
        alt,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )

def test_spatial_20_grid_mi_agenda():
    import gridvoting as gv
    np = gv.np
    xp = gv.xp
    grid = gv.Grid(x0=-20,x1=20,y0=-20,y1=20)
    number_of_alternatives = 41*41
    assert len(grid.x) == number_of_alternatives
    assert len(grid.y) == number_of_alternatives
    on_grid_boundary = (grid.x == -20) | \
                    (grid.x == 20) | \
                    (grid.y == -20) | \
                    (grid.y == 20)
    assert on_grid_boundary.shape == (number_of_alternatives,)
    voter_ideal_points = np.array([
        [-15,-9],
        [0,17],
        [15,-9]
    ])
    number_of_voters = 3
    assert voter_ideal_points.shape == (3,2)
    majority = 2
    u = grid.spatial_utilities(
        voter_ideal_points=voter_ideal_points,
        metric='sqeuclidean'
    )
    assert u.shape == (number_of_voters, number_of_alternatives)
    vm = gv.VotingModel(
        utility_functions=u,
        majority=majority,
        zi=False,
        number_of_voters=number_of_voters,
        number_of_feasible_alternatives=number_of_alternatives
    )
    vm.analyze()
    stat_dist = vm.stationary_distribution
    assert abs(stat_dist.sum() - 1)<1e-9
    assert abs(sum(stat_dist[on_grid_boundary])- 0.024) < 0.001
    stat_dist_gz = stat_dist[stat_dist>0.0]
    # entropy == 10.32 from Brewer,Juybari,Moberly(2023), Tab 5, Row 20-MI
    assert abs(-np.sum(stat_dist_gz*np.log2(stat_dist_gz))-10.32) < 0.01
    stat_dist_algebraic = xp.asnumpy(vm.MarkovChain.solve_for_unit_eigenvector())
    assert abs(np.sum(np.abs(stat_dist-stat_dist_algebraic))) < 1e-9
