#Ricci Flow Neural Network
Framework for using Physics Informed Neural networks to get metric solution to Ricci Flow equation as a function of time.
Have written for 2D torus geometry, and the 2D cigar soliton. The geometries are completely specified within the initial metric
loss functions of the NN. Terms corresponding to boundary conditions/symmetry conditions are also included in the loss function.
In order to fit any geometry, a simple redefinition of the initial and symmetry loss terms is required to suit the required metric.
