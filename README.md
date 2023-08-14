# gridvoting

This software is a library module for our [research publication (open-access)](https://doi.org/10.1007/s11403-023-00387-8):
<pre>
  Brewer, P., Juybari, J. & Moberly, R. 
  A comparison of zero- and minimal-intelligence agendas in majority-rule voting models. 
  J Econ Interact Coord (2023). https://doi.org/10.1007/s11403-023-00387-8
</pre>

This software helps set up, calculate, and plot stationary probability distributions for
sequential voting simulations (with ZI or MI random challengers) that take place on a 2D grid of feasible outcomes.

In our paper, we used the simulations to show that adding intelligence to the agenda of a collection of voting bots does not necessarily
improve the fairness or reasonableness of outcomes.  We're not claiming adding intelligence is always bad, since one 
cannot deduce such generalities from a few simulations. But in some well-known scenarios, the simulations demonstrate cases where
adding intelligence to the voting agenda can increase the variance and decrease the equality of outcomes for equally situated agents.

## License

The software is provided under the standard [MIT License](./LICENSE.md). 

You are welcome to try the software, read it, copy it, adapt it to your
needs, and redistribute your adaptations. If you change the software, be sure to change the module name somehow so that
others know it is not the original.  See the LICENSE file for more details.  

## Disclaimers

The software is provided in the hope that it may be useful to others, but it is not a full featured turnkey
system for conducting arbitrary voting simulations. Additional coding is required to define a specific simulation.
Also, we do not include an automated test suite because our free testing provider does not include a GPU in the test environment.  
Some manual testing has been done, but is not extensive.  The software may contain bugs, unexpected behaviors, or have other issues.

The [MIT License](./LICENSE.md) also includes this disclaimer: 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## code and data for specific simulations

Code specific to the spatial voting and budget voting portions of our research publication above -- as well as output data -- 
is deposited at: [OSF Dataset for A comparison of zero and minimal Intelligence agendas in majority rule voting models](https://osf.io/k2phe/)
and is freely available.

## use of Google Colab

We used [Google Colab](https://colab.google), a cloud-based service that runs Python-based analyses on Google's servers and GPUs,
for conducting most of the research reported in the publication above.  When using Google Colab, the local computer does NOT need to have a GPU.

The software has also run (without Colab) on a local computer with a Nvidia gaming GPU, and remote computers with industrial Nvidia A100 GPUs.

## requirements
* Nvidia GPU with minimum of 16GB GPU memory to duplicate simulations reported in the above paper
* Nvidia CUDA drivers (except on Google Colab, where CUDA is pre-installed)
* Python 3 
* **all** of these Python-3 scientific computing modules (all except cupy are pre-installed on Google Colab, and [cupy can be installed from these instructions](https://docs.cupy.dev/en/stable/install.html)):
  - numpy
  - pandas
  - matplotlib
  - scipy
  - cupy
* familiarity with Python language / scientific computing / gpu Nvidia-CUDA setup

## Random sequential voting simulations

This follows [section 2 of our research paper](https://link.springer.com/article/10.1007/s11403-023-00387-8#Sec4)

A simulation consists of a sequence of times: `t=0,1,2,3,...`
a finite feasible set of alternatives **F**, a set of voters who have preferences over the alternatives and vote truthfully,
a rule for voting and selecting challengers, and a mapping of the set of alternativies **F** into a 2D grid.  

The active or status quo alternative at time t is called `f[t]`.  

At each t, there is a majority-rule vote between alternative `f[t]` and a challenger
alternative `c[t]`.  The winner of that vote becomes the next status quo `f[t+1]`.  

Randomness enters the simulation through two possible rules for choosing the challenger
`c[t]`.  When `zi=True`, `c[t]` is chosen from the "Zero Intelligence" agenda which consists
of a uniform random distribution over **F**.  When `zi=False`, `c[t]` is chosen from the 
"Minimal Intelligence" agenda which is a uniform distribution over the status quo `f[t]` and the possible
winning alternatives given the status quo `f[t]`.

## Specialization of parts of the software / Separation of Concerns

The `VotingModel` class manages simulations.  Each simulation is an instance of `VotingModel`.  The constructor
requires various properties of the simulation, such as the number of voters, number of alternatives, and voters' utility functions.
Utility functions are arrays of dim `(number_of_voters, number_of_alternatives)` defining the preferences of each voter for
each outcome as a number where more is better. It is the ordering and not the values that are important.  
Besides plotting and answering simple questions about the alternatives, the class also provides code
for calculating the transition matrix and providing it as an input to the `MarkovChainGPU` class below.

The `Grid` class manages rectangular grids. An instance of `VotingModel` will usually specify
a Grid instance for defining utility functions and plotting/visualization purposes.  It is also possible to use `VotingModel`
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
