# Ricci-flow-neural-network
Use universal approximation theorem of Neural network to approximate a metric tensor
as a Neural network. Now use the ‘Physics informed neural network’ framework to solve
the non linear Ricci flow PDE to evolve the metric to a flat metric. This works for
any general metric geometry and can be specialised by changing the initial conditions.

Framework for using Physics Informed Neural networks to get metric solution to Ricci Flow equation as a function of time.
Have written for 2D torus geometry, and the 2D cigar soliton. The geometries are completely specified within the initial metric
loss functions of the NN. Terms corresponding to boundary conditions/symmetry conditions are also included in the loss function.
In order to fit any geometry, a simple redefinition of the initial and symmetry loss terms is required to suit the required metric.
