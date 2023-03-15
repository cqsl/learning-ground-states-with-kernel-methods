This folder contains wrappers around jax' linear solvers

and modified versions of several iterative solvers which differently from their jax counterparts:
- are MPI aware (via mpi4jax)
- return solver stats, i.e. number of iterations performed etc
