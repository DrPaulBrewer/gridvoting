# gridvoting

This software helps set up, calculate, and plot stationary probability distributions for
sequential voting simulations (with ZI or MI random challengers) that take place on a 2D grid of feasible outcomes.

This software generated data for our [research publication](https://doi.org/10.1007/s11403-023-00387-8):
<pre>
  Brewer, P., Juybari, J. & Moberly, R. 
  A comparison of zero- and minimal-intelligence agendas in majority-rule voting models. 
  J Econ Interact Coord (2023). https://doi.org/10.1007/s11403-023-00387-8
</pre>

You are welcome to try the software, read it, copy it, and adapt it to your
needs. If you change the software, be sure to change the module name somehow so that
others know it is not the original.  See the LICENSE file for more details.  

The software is provided in the hope that it may be useful to others, but it is not a full featured turnkey
system for conducting arbitrary voting simulations. While some manual tests have been done, 
there are currently no automated tests.  

## requirements
* modern NVIDIA GPU: either a remote Google CoLab GPU or a local GPU on the computer running the module
* GPU needs 16GB or more of GPU memory to duplicate simulations reported in the above paper
* Python 3 
* **all** of these Python 3 scientific computing modules (all except cupy are on Google Colab, and cupy can be installed):
      - numpy
      - cupy
      - pandas
      - matplotlib
      - scipy
* Nvidia CUDA drivers (except on Google Colab, where CUDA is pre-installed)
* familiarity with Python language / scientific computing / gpu Nvidia-CUDA setup

## Random sequential voting simulations

A simulation consists of a sequence of times: `t=0,1,2,3,...`
a finite feasible set of alternatives **F**, a rule for selecting challengers,
and a mapping of the set of alternativies **F** into a 2D grid.  

The active or status quo alternative at time t is called `f[t]`.  

At each t, there is a majority-rule vote between alternative `f[t]` and a challenger
alternative `c[t]`.  The winner of that vote becomes the next status quo `f[t+1]`.  

Randomness enters the simulation through two possible rules for choosing the challenger
`c[t]`.  When `zi=True`, `c[t]` is chosen from the "Zero Intelligence" agenda which consists
of a uniform random distribution over **F**.  When `zi=False`, `c[t]` is chosen from the 
"Minimal Intelligence" agenda which is a uniform distribution over the status quo `f[t]` and the possible
winning alternatives given the status quo `f[t]`.

## Specialization of parts of the software / Separation of Concerns

The `VotingModel` class manages simulations.  Each simulation is an instance of `VotingModel`.
Besides plotting and simple questions about the alternatives, the class also provides code
for calculating the transition matrix needed as an input to the `MarkovChainGPU` class below.

The `Grid` class manages rectangular grids. An instance of `VotingModel` will usually specify
an instance of grid for plotting or visualization purposes, but it is not strictly necessary.

The `MarkovChainGPU` class manages a Markov Chain calulation on a GPU.  This class is called 
internally from `VotingModel`.  The class contains two methods for calculating the 
stationary distribution of a Markov Chain: the power method (default), and an algebraic method
(optional).  

## WORK IN PROGRESS

## Old text

Inputs:
* grid size in 2D -glimit <= x,y <= glimit
* utility functions for each voter for each point on the grid
* challenger strategy (zi True/False)

Outputs:
* Markov chain transition matrix
* Existence of core (absorbing) points
* Stationary distributions (no core)
* Diagnostic and distribution plots
