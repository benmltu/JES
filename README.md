# Joint Entropy Search (JES)

This repository contains the code for some multi-objective information-theoretic acquisition functions: Joint Entropy Search (JES), Max-value Entropy Search (MES) and Predictive Entropy Search (PES). All of the acquisition functions are implemented for the BoTorch library (https://github.com/pytorch/botorch/).

This code was initially implemented with the following dependencies:

- Python 3.8
- BoTorch 0.5.1
- PyTorch 1.9.0
- GPytorch 1.6
- Pymoo 0.5.0
- SciPy 1.7.3
- NumPy 1.21.2

In the notebooks folder I have demonstrated how to use the main methods included in this repository. If you want to discuss more about the code presented here feel free to raise a discussion or to e-mail me ben.tu16@imperial.ac.uk.

### Additional notes

- The acquisition functions should be able to handle inequality constraints. To utilize the functionality, one would have to implement a constrained multi-objective solver, which returns an approximation to the _feasible_ Pareto set and front. This functionality works for the JES and MES acquisition function because the constraints only appear when computing the box-decompositions. The PES acquisition function might cause some issues because it relies on additional hardcoded calculations that have not been tested extensively.

- The PES acquisition function is approximated using expectation propagation. The automatic gradients inferred for the PES are questionable in practice because certain operations used in the procedure are not differentiable. Therefore, we suggest that this acquisition function should be optimized using gradients approximated by finite differences.
