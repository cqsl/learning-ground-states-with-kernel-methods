This folder contains a modified version of netket.stats,
which threads through xla tokens needed to avoid deadlocks when used inside a jitted function with MPI via mpi4jax  (we have seen them only on gpu).
