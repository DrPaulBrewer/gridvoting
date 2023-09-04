import gridvoting as gv
def test_gridvoting_topcycle():
  np = gv.np
  xp = gv.xp
  u = np.array([
    [1000,900,800,20,10,1],
    [800,1000,900,1,20,10],
    [900,800,1000,10,1,20]
  ])
  vm = gv.VotingModel(utility_functions=u,number_of_feasible_alternatives=6,number_of_voters=3,majority=2,zi=False)
  vm.analyze()
  xp.testing.assert_array_almost_equal(
    vm.stationary_distribution,
    [1/3,1/3,1/3,0.,0.,0.],
    1e-9
  )
  xp.testing.assert_array_equal(
    vm.stationary_distribution[3:],
    [0.,0.,0.]
  )
  
