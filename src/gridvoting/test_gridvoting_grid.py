import pytest

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
    assert grid.boundary[0]
    assert grid.boundary[-1]
    assert grid.boundary.shape == (165,)
    correct_boundary = np.array([
    [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False,  False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, False, False, True],
    [True,   True,  True,  True, True,  True,  True,  True,  True,  True,  True]
    ])
    np.testing.assert_array_equal(grid.boundary.reshape(grid.gshape), correct_boundary)
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

def test_grid_embedding():
    import gridvoting as gv
    np = gv.np
    grid = gv.Grid(x0=-5,x1=5,y0=-7,y1=7)
    triangle = (grid.x>=0) & (grid.y>=0) & ((grid.x+grid.y)<=4)
    correct_triangle = np.array([
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False,  True, False, False, False, False, False],
    [False, False, False, False, False,  True,  True, False, False, False, False],
    [False, False, False, False, False,  True,  True,  True, False, False, False],
    [False, False, False, False, False,  True,  True,  True,  True, False, False],
    [False, False, False, False, False,  True,  True,  True,  True,  True, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False, False]
    ])
    np.testing.assert_array_equal(triangle.reshape(grid.gshape),correct_triangle)
    assert 15 == sum(triangle)
    triangle_points_xy = grid.as_xy_vectors()[triangle]
    correct_triangle_points_xy = np.array([
        [0,4],
        [0,3],
        [1,3],
        [0,2],
        [1,2],
        [2,2],
        [0,1],
        [1,1],
        [2,1],
        [3,1],
        [0,0],
        [1,0],
        [2,0],
        [3,0],
        [4,0]
    ])
    emfunc = grid.embedding(valid=triangle)
    triangle_x = grid.x[triangle]
    np.testing.assert_array_equal(
        triangle_x,
        np.array([0,0,1,0,1,2,0,1,2,3,0,1,2,3,4])
    )
    correct_embedding_result = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    np.testing.assert_array_equal(
        emfunc(triangle_x, fill=0.0).reshape(grid.gshape),
        correct_embedding_result
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


