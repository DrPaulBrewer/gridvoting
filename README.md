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

## use of Google Colab

We used [Google Colab](https://colab.google), a cloud-based service for running Python analysis notebooks on Google's servers and GPUs,
for conducting most of the research reported in the publication above.  When using Google Colab, the local computer does NOT need to have a GPU.

The software has also been tested (without Colab) on a local computer with a Nvidia gaming GPU, and remote computers with industrial Nvidia A100 GPUs.

## requirements
* NVIDIA GPU with minimum of 16GB GPU memory to duplicate simulations reported in the above paper
* Python 3 
* **all** of these Python 3 scientific computing modules (all except cupy are pre-installed on Google Colab, and [cupy can be installed from these instructions](https://docs.cupy.dev/en/stable/install.html)):
  - numpy
  - pandas
  - matplotlib
  - scipy
  - cupy
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
for calculating the transition matrix and providing it as an input to the `MarkovChainGPU` class below.

The `Grid` class manages rectangular grids. An instance of `VotingModel` will usually specify
a Grid instance for plotting or visualization purposes.  It is also possible to use `VotingModel`
without specifying any kind of grid or coordinate mapping, for an example see class `CondorcetCycle`.

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
