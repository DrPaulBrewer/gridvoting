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
 

def test_grid_as_xy_vectors():
    import gridvoting as gv
    np = gv.np
    grid = gv.Grid(x0=-3,x1=3,y0=-5,y1=5)
    correct_vectors = np.array([
       [-3,  5],
       [-2,  5],
       [-1,  5],
       [ 0,  5],
       [ 1,  5],
       [ 2,  5],
       [ 3,  5],
       [-3,  4],
       [-2,  4],
       [-1,  4],
       [ 0,  4],
       [ 1,  4],
       [ 2,  4],
       [ 3,  4],
       [-3,  3],
       [-2,  3],
       [-1,  3],
       [ 0,  3],
       [ 1,  3],
       [ 2,  3],
       [ 3,  3],
       [-3,  2],
       [-2,  2],
       [-1,  2],
       [ 0,  2],
       [ 1,  2],
       [ 2,  2],
       [ 3,  2],
       [-3,  1],
       [-2,  1],
       [-1,  1],
       [ 0,  1],
       [ 1,  1],
       [ 2,  1],
       [ 3,  1],
       [-3,  0],
       [-2,  0],
       [-1,  0],
       [ 0,  0],
       [ 1,  0],
       [ 2,  0],
       [ 3,  0],
       [-3, -1],
       [-2, -1],
       [-1, -1],
       [ 0, -1],
       [ 1, -1],
       [ 2, -1],
       [ 3, -1],
       [-3, -2],
       [-2, -2],
       [-1, -2],
       [ 0, -2],
       [ 1, -2],
       [ 2, -2],
       [ 3, -2],
       [-3, -3],
       [-2, -3],
       [-1, -3],
       [ 0, -3],
       [ 1, -3],
       [ 2, -3],
       [ 3, -3],
       [-3, -4],
       [-2, -4],
       [-1, -4],
       [ 0, -4],
       [ 1, -4],
       [ 2, -4],
       [ 3, -4],
       [-3, -5],
       [-2, -5],
       [-1, -5],
       [ 0, -5],
       [ 1, -5],
       [ 2, -5],
       [ 3, -5]
    ])
    np.testing.assert_array_equal(
        grid.as_xy_vectors(),
        correct_vectors
    )

@pytest.mark.parametrize("x0,x1,xstep,y0,y1,ystep,correct",[
    (None,None,None,None,None,None,(10,6)),
    (0,0,None,0,0,None,(1,1)),
    (-5,5,None,-20,20,2,(21,11)),
    (1,4,None,None,None,None,(10,4))
])
def test_grid_shape(x0,x1,xstep,y0,y1,ystep,correct):
    import gridvoting as gv
    grid = gv.Grid(x0=0,x1=5,y0=0,y1=9)
    assert grid.shape(x0=x0,x1=x1,xstep=xstep,y0=y0,y1=y1,ystep=ystep) == correct

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
    

@pytest.mark.parametrize("zi,correct_P", [
    (True, [
        [2./3.,0.,1./3.],
        [1./3.,2./3.,0.],
        [0., 1./3., 2./3.]
    ]),
    (False,[
        [ 1./2., 0, 1./2.],
        [ 1./2., 1./2., 0],
        [ 0,  1./2., 1./2.]
    ])
])
def test_condorcet(zi, correct_P):
    import gridvoting as gv
    xp = gv.xp
    condorcet_model =  gv.CondorcetCycle(zi=zi)
    assert not condorcet_model.analyzed
    condorcet_model.analyze()
    assert condorcet_model.analyzed
    mc = condorcet_model.MarkovChain
    gv.xp.testing.assert_array_almost_equal(
        mc.P,
        xp.array(correct_P),
        decimal=10
    )
    gv.xp.testing.assert_array_almost_equal(
        condorcet_model.stationary_distribution,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )
    mc=condorcet_model.MarkovChain
    alt = mc.solve_for_unit_eigenvector()
    gv.xp.testing.assert_array_almost_equal(
        alt,
        xp.array([1.0/3.0,1.0/3.0,1.0/3.0]),
        decimal=10
    )

# attempt to replicate grid boundary probability and entropy (H) from 
# Brewer, Juybari, Moberly (2023), J. Econ Interact Coord, Tab.5
# grid size 20 only
# https://link.springer.com/article/10.1007/s11403-023-00387-8/tables/5
@pytest.mark.parametrize("params,correct", [
    ({'g':20,'zi':False}, {'p_boundary': 0.024, 'entropy': 10.32}),
    ({'g':20,'zi':True},  {'p_boundary': 0.0086,'entropy':  9.68})
])
def test_replicate_spatial_voting_analysis(params, correct):
    import gridvoting as gv
    np = gv.np
    xp = gv.xp
    g = params['g']
    zi = params['zi']
    majority = 2
    grid = gv.Grid(x0=-g,x1=g,y0=-g,y1=g)
    number_of_alternatives = (2*g+1)*(2*g+1)
    assert len(grid.x) == number_of_alternatives
    assert len(grid.y) == number_of_alternatives
    on_grid_boundary = (grid.x == -g) | \
                    (grid.x == g) | \
                    (grid.y == -g) | \
                    (grid.y == g)
    assert on_grid_boundary.shape == (number_of_alternatives,)
    voter_ideal_points = np.array([
        [-15,-9],
        [0,17],
        [15,-9]
    ])
    number_of_voters = 3
    assert voter_ideal_points.shape == (3,2)
    u = grid.spatial_utilities(
        voter_ideal_points=voter_ideal_points,
        metric='sqeuclidean'
    )
    assert u.shape == (number_of_voters, number_of_alternatives)
    vm = gv.VotingModel(
        utility_functions=u,
        majority=majority,
        zi=zi,
        number_of_voters=number_of_voters,
        number_of_feasible_alternatives=number_of_alternatives
    )
    vm.analyze()
    stat_dist = vm.stationary_distribution
    assert stat_dist.sum() == pytest.approx(1.0,abs=1e-9)
    p_boundary = sum(stat_dist[on_grid_boundary])
    assert p_boundary == pytest.approx(correct['p_boundary'], rel=0.05)
    stat_dist_gz = stat_dist[stat_dist>0.0]
    entropy = -stat_dist_gz.dot(np.log2(stat_dist_gz))
    assert entropy == pytest.approx(correct['entropy'], abs=0.01)
    stat_dist_algebraic = xp.asnumpy(vm.MarkovChain.solve_for_unit_eigenvector())
    l1_power_vs_algebraic_solns = np.linalg.norm(stat_dist-stat_dist_algebraic, ord=1)
    assert l1_power_vs_algebraic_solns < 1e-9
