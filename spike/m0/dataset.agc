## The one input of the M0 spike.
##
## A fixed batch, fixed initial parameters, and the step count, learning rate
## and tolerance of the comparison, named in one place so that the demo, the
## benchmark and the tests cannot disagree about them. Nothing here is
## random: two runs of the demo print the same curve, which is what makes its
## output evidence.

## The batch: 4 rows of 3 features, row-major.
X_SHAPE: List[Int] = [4, 3]
X_DATA: List[Float] = [1.0, -0.5, 0.25, 0.5, 1.0, -0.75, -0.25, 0.5, 1.0, 1.0, 1.0, -1.0]

## The parameters to start from: 3 features in, 2 out.
W_SHAPE: List[Int] = [3, 2]
W_DATA: List[Float] = [0.4, -0.6, 0.2, 0.55, -0.7, 0.35]

## Twenty steps at a learning rate small enough that the piecewise-linear
## loss falls at every one of them: a larger one steps across a relu kink and
## the curve stops being monotone, which says nothing about the gradient.
STEPS = 20
LR = 0.02

## How far apart the reference and libtorch may be and still count as the
## same curve. The reference computes in Agentica `Float` (IEEE 754 double)
## and libtorch in `float32`, so the forward pass is not the same arithmetic;
## the measured gap is about 1.5e-7 (see README).
TOLERANCE = 0.000001
