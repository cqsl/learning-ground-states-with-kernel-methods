# learning-ground-states-with-kernel-methods


## Installation

```
python3 -m pip install git+https://github.com/cqsl/learning_ground_states_with_kernel_methods.git
```
It is advisable to install inside a virtual environment as the version of the dependencies (jax and netket) is hardcoded in the setup script.
To enable the optional MPI support `mpi4jax` needs to be installed manually.

## Simulations

The main simulation script for finding ground states is `slpm/ground_state/gs.py`, which is installed as `slpm_gs`.
The documentation of its command-line arguments can be accessed with `slpm_gs -h`

## Examples

### TFI 1D
Run 100 steps of the SLPM for the TFI in 1D with 16 Spins at h=1, with a dataset of size 512, sampling from 32 chains in parallel, using Lambda=1, saving every 10 steps to /tmp/foo_*.pkl :
```python
slpm_gs -n 100 -m tfi -d 1 -L 16 -H 1 -N 512 -c 32  -l 1 -s 10 -o /tmp/foo
```
where the final output will be saved in `/tmp/foo_0099_rank_0_of_1.pkl`.

The script can also be used to sample from the final state using a large number of samples to accurately estimate the energy.
The parameters ar similar as before, except that now we only need 1 step, we take 64K samples and restart our simulations from `/tmp/foo_0099_rank_0_of_1.pkl`. Furthermore we tell the script to compute both the kernel matrix and estimate the energies in blocks of 256 samples, to avoid OOM.
```python
slpm_gs -n 1 -m tfi -d 1 -L 16 -H 1 -N 65536 -c 32  -l 1 -s 10 -o /tmp/foo_final -i /tmp/foo_0099_rank_0_of_1.pkl -bsk 256 -bse 256
```
