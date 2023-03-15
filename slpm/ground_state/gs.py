import argparse
import sys

import numpy as np
import math
import netket as nk
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial
import tqdm
import pickle


from slpm.solvers import *
from slpm.operators import *
from slpm.exact import ising1d_energy
from slpm.sampling import *
from slpm.sampling.unique.utils import *
from slpm.kernels import simple_kernel_fn
from slpm.utils import *
from slpm.step import *
from slpm.ground_state.reference_energies import *


parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="input pkl to continue from", required=False)
parser.add_argument("-o", type=str, help="output prefix", required=True)
parser.add_argument("-s", type=int, help="save interval", default=1)

parser.add_argument("-l", type=float, help="lambda", required=True)
parser.add_argument("-n", type=int, help="steps", required=True)

parser.add_argument("-L", type=int, help="# spins per dim", required=True)
parser.add_argument("-d", type=int, help="# dimensions", default=1)

parser.add_argument("-N", type=int, help="samples per rank", required=True)
parser.add_argument("-c", type=int, help="chains per rank", required=True)

parser.add_argument("-m", type=str, help="model (specify tfi or afh)", required=True)
parser.add_argument("-H", type=float, help="tfi transverse field strength", required=False)

parser.add_argument("-dr", type=float, help="diagonal regularization", default=1e-4)

parser.add_argument("--ker", type=str, help="name of kernel (ntk_rbmsymm, ...)", default="ntk_rbmsymm")
parser.add_argument("--seed", type=int, help="seed", default=27)

parser.add_argument("--iterative", type=str, help="use an iterative solver instead of cholesky, choose between cg/gmres/bicgstab")
parser.add_argument("--maxiter", type=float, help="maxiter for iterative solvers", default=1000)
parser.add_argument("--svd", action="store_true", help="use svd instead of cholesky; please specify svdcutoff")
parser.add_argument("--svdcutoff", type=float, help="singular value cutoff")

parser.add_argument(
    "--sample_unique", action="store_true", help="remove duplicate samples; the defualt is to sample N uniuque samples, otherwise consider passing --const_n_samp; also see uniq_weight_diag_reg"
)
parser.add_argument("--uniq_weight_diag_reg", action="store_true", help="per-sample regularization, using the inverse of the number of occurences")
parser.add_argument("--const_n_samp", help="sample a constant number, and extract unique ones", action="store_true")
parser.add_argument(
    "-ni_ratio",
    type=float,
    help="ratio for unique sampling; how many samples to sample extra at each step until there are enough unique ones as a fraction of N; only relevant if not using const_n_samp",
    default=1.0,
)

parser.add_argument("--sg", help="full space group not just translation group", action="store_true")

parser.add_argument("--realspace", help="learn psi instead of logpsi", action="store_true")
parser.add_argument("--notbehind", action="store_true", help="sampling from the new state and not from the old one")

parser.add_argument("--save_full_statevec", action="store_true", help="predict the full state vector; useful for testing on small systems")

parser.add_argument("-bse", type=int, help="eloc computation blocksize; use if you run into OOM; e.g. 256 seems to be a safe value", default=None)
parser.add_argument("-bsk", type=int, help="kernel computation blocksize; use if you run into OOM; e.g. 256 seems to be a safe value", default=None)
parser.add_argument("-bsl", type=int, help="full state vector prediction blocksize (only applies if --save_full_statevec is specified); use if you run into OOM; e.g. 256 seems to be a safe value", default=None)

group = parser.add_mutually_exclusive_group()
group.add_argument("-E0", type=float, help="pass E0 for comparison if not present in reference_energies")
group.add_argument("-E0N", type=float, help="pass E0 per site for comparison")






def main(argv=None):
    if argv is None:
        argv = sys.argv
    args = parser.parse_args(argv[1:])
    print(vars(args))

    ###############################################################################
    # GENERAL SETUP
    ###############################################################################
    L = args.L
    n_dim = args.d  # 1D L, 2D LxL etc
    N = args.L**args.d  # number of sites
    h = args.H  # transverse field strength
    lambd = args.l
    model = args.m

    # per rank
    N_samples_per_rank = args.N
    n_chains_per_rank = args.c
    if not args.sample_unique:
        assert n_chains_per_rank <= N_samples_per_rank

    sample_unique = args.sample_unique

    if (nk.utils.mpi.n_nodes > 1 and sample_unique):
        raise NotImplementedError('unique sampling is not implemented with mpi yet')

    chain_length = max(N_samples_per_rank // n_chains_per_rank, 1)
    n_discard = max(chain_length // 8, 1)


    seed = args.seed

    # dtype = jnp.float32 # for X
    dtype = jnp.float64  # for X

    n_steps = args.n
    i0 = 0  # 1st step
    save_interval = args.s
    diag_reg = args.dr
    actual_diag_reg = diag_reg


    if args.uniq_weight_diag_reg:
        # already set the correct shape here to avoid recompilation
        diag_reg = jnp.ones(N_samples_per_rank) * actual_diag_reg

    block_size_logpsi = args.bsl

    ###############################################################################
    # SOLVER SETUP
    ###############################################################################
    iterative = args.iterative is not None
    if iterative:
        assert not args.svd

    iterative_solve_fn = None
    get_solve_fn = None
    if args.svd:
        assert args.iterative is None
        assert args.svdcutoff is not None  # please specify the svd cutoff
        get_solve_fn = Partial(get_svd_solve, cutoff=args.svdcutoff, diag_reg=diag_reg)
    elif args.iterative == "cg":
        iterative_solve_fn = Partial(cg)
    elif args.iterative == "gmres":
        # iterative_solve_fn = Partial(partial(gmres, solve_method='incremental'))
        iterative_solve_fn = Partial(partial(gmres, solve_method="blocked"))
    elif args.iterative == "bicgstab":
        iterative_solve_fn = Partial(bicgstab)
    elif args.iterative is None:
        get_solve_fn = Partial(get_cho_solve, diag_reg=diag_reg)
    else:
        assert False

    if nk.utils.mpi.n_nodes > 1 and not iterative:
        raise ValueError('mpi requires the use of an iterative solver; please use --iterative ...')

    ###############################################################################
    # RNG SETUP
    ###############################################################################


    token = jax.lax.create_token()

    # mpi-aware
    rng_key = nk.jax.PRNGKey(seed)
    rng_key = nk.jax.mpi_split(rng_key)


    ###############################################################################
    # MODEL SETUP
    ###############################################################################

    g = nk.graph.Hypercube(L, n_dim, pbc=True)
    edges = jnp.array(g.edges())

    E0 = args.E0

    if args.E0N is not None:
        E0 = args.E0N * hi.size

    if model == "tfi":
        hi = nk.hilbert.Spin(1 / 2, g.n_nodes)
        sa = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=n_chains_per_rank, dtype=dtype)
        if h is None:
            raise ValueError('please specify the transverse field strength with -H ...')
        ha = get_ising_kernel_jax(edges, h, J=1)
        if E0 is None:
            if n_dim == 1:
                E0 = ising1d_energy(L, h)
            elif n_dim == 2:
                try:
                    E0 = E0_ref["tfi", n_dim, L]
                except KeyError:
                    pass
    elif model == "afh":
        hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)
        sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains_per_rank=n_chains_per_rank, dtype=dtype)
        assert h is None
        ha = get_heisenberg_kernel_jax(edges, Jz=1, sign_rule=True)
        if E0 is None:
            try:
                E0 = E0_ref["afh", n_dim, L]
            except KeyError:
                pass
    # elif model == 'j1j2':
    # TODO re-add j1j2
    else:
        raise ValueError("unknown model")

    if E0 is None:
        if nk.utils.mpi.rank == 0:
            print("No reference energy available, please specify E0 or E0N if needed")
    else:
        if nk.utils.mpi.rank == 0:
            print(f"E0 = {E0:0.5f}")
            print(f"E0/N = {E0/hi.size:0.5f}")

    k, rng_key = jax.random.split(rng_key, 2)
    sampler_state = get_sampler_state(sa, k)

    ###############################################################################
    # PROPAGATOR SETUP
    ###############################################################################

    block_size_eloc = args.bse
    lvk = local_val_kernel_log_lambda
    if block_size_eloc is not None:
        lvk = block_local_val_kernel(lvk.__wrapped__, block_size_eloc)
        lvk = jax.jit(lvk, inline=True, static_argnames=("return_logpsi_x", "sgn_fn", "return_eloc"))

    op = Partial(lvk, lambd=lambd, op_kernel=ha)

    ###############################################################################
    # SYMMETRIES SETUP
    ###############################################################################
    if args.sg:
        sg = g.space_group()
    else:
        sg = g.translation_group()

    G = np.array(sg)


    def _symm_repr(G, x):
        return shiftall_mat(x, G, spin_flip_symm=True, ret_int=True)


    # function which finds the representative of a configuration, using the minimum integer convention
    # returns the integer representation
    symm_repr = Partial(_symm_repr, G)

    ###############################################################################
    # INITIAL STATE SETUP
    ###############################################################################

    if args.save_full_statevec:
        x_test = jnp.array(hi.all_states()).astype(dtype)
    else:
        x_test = None

    k, rng_key = jax.random.split(rng_key, 2)

    if args.sample_unique:
        multiplicator = 10
        n, r = divmod(N, 64)
        lengths = (64,) * n
        if r != 0:
            lengths = (r,) + lengths
        k, k_, *k__ = jax.random.split(k, len(lengths) + 2)
        ii = [jax.random.randint(k___, (N_samples_per_rank * multiplicator,), 0, jnp.array(2**l - 1, dtype=jnp.uint64), dtype=jnp.uint64) for l, k___ in zip(lengths, k__)]
        if len(lengths) <= 1:
            i = ii[0]
        else:
            i = jnp.array(ii).T
        iu = jnp.unique(i, axis=0)
        iu = jax.random.permutation(k_, iu, independent=True)
        if not len(iu) >= N_samples_per_rank:
            raise ValueError("failed to generate enought initial unique samples; try increasing the multiplicator")
        iu = iu[:N_samples_per_rank]
        x_train = int2vec(iu, N).astype(dtype)
    else:
        x_train = jnp.array(hi.random_state(k, (N_samples_per_rank,))).astype(dtype)

    y_train = jnp.zeros((N_samples_per_rank,))

    ###############################################################################
    # INPUT/OUTPUT SETUP
    ###############################################################################

    out_prefix = args.o

    # code to load an existing run and continue
    # the new number of chains needs to be the same as the old one
    if args.i is not None:
        fname = args.i
        if nk.utils.mpi.n_nodes > 1:
            fname = fname.split("_rank_")[0]
            fname += f"_rank_{nk.utils.mpi.rank}_of_{nk.utils.mpi.n_nodes}.pkl"
        print("loading", fname)
        with open(fname, "rb") as f:
            out = pickle.load(f)
        x_train = out["x_train_new"]
        y_train = out["y_train_new"]
        i0 = out["step"] + 1
        energies_so_far = out["energies_so_far"]
        sampler_state = out["sampler_state"]
        if (out.get("counts", None) is not None) and args.uniq_weight_diag_reg:
            diag_reg = actual_diag_reg / out["counts"]

    if args.const_n_samp:
        step_fn = step_dynamic
    else:
        step_fn = step

    ###############################################################################
    # KERNEL SETUP
    ###############################################################################


    if args.ker == "ntk_rbmsymm":

        gamma = 0.5808428804676404

        def f(gamma, x):
            return jnp.arcsin(gamma * x) * x

        kernel_fn = Partial(nk.jax.HashablePartial(simple_kernel_fn, L=hi.size, out_dtype=jnp.float64), G=G, f=Partial(f, gamma))
    else:
        raise ValueError("unknown kernel")

    block_size_kernel = args.bsk

    if block_size_kernel is not None:
        kernel_fn = block_kernel_fn(kernel_fn, block_size=block_size_kernel)


    ###############################################################################
    # MAIN LOOP
    ###############################################################################
    energies_so_far = []
    y_train_prev = y_train

    pbar = tqdm.trange(i0, i0 + n_steps, disable=nk.utils.mpi.rank != 0)
    for i in pbar:

        # step

        res = step_fn(
            kernel_fn,
            x_train,
            y_train,
            y_train_prev,
            sa,
            sampler_state,
            chain_length,
            N_samples_per_rank,
            n_discard,
            get_solve_fn,
            iterative_solve_fn,
            diag_reg,
            iterative,
            sample_unique,
            args.notbehind,
            args.realspace,
            op,
            symm_repr,
            x_test,
            args.ni_ratio,
            args.maxiter,
            block_size_logpsi,
            token,
        )
        x_train_new, y_train_new, y_train_prev, sampler_state, eloc, stats, solver_stats, counts, sampler_stats, y_test, shift, token = res

        # stats

        if counts is not None and not args.const_n_samp:
            stats = nk.stats.Stats((eloc.ravel() * counts).sum() / counts.sum())

        stats_str = str(stats).ljust(48)
        descr = f"E={stats_str} E/N={stats.mean/hi.size:0.5f}"
        if E0 is not None:
            err = float(stats.mean) - E0
            err_relative = jnp.abs(err / E0)
            err_str = f"{err:0.2e}".ljust(9)
            descr += f" Err={err_str}  Erel={err_relative:0.2e}"
            if sampler_stats is not None:
                n_unique, _, _, n_total = sampler_stats
                descr += f" u.samp: {int(n_unique)}/{n_total}"
        pbar.set_description(descr)

        # output

        out = {}
        out["vars"] = vars(args)
        out["step"] = i
        out["rank"] = nk.utils.mpi.rank
        out["n_ranks"] = nk.utils.mpi.n_nodes
        out["shift"] = shift
        out["counts"] = counts
        out["sampler_stats"] = sampler_stats

        # before
        out["x_train"] = x_train
        out["y_train"] = y_train
        # after
        out["x_train_new"] = x_train_new
        out["y_train_new"] = y_train_new
        #
        out["stats"] = stats
        out["eloc"] = eloc
        out["sampler_state"] = sampler_state
        out["lambd"] = lambd
        out["y_test"] = y_test

        if solver_stats is not None:
            out["solver_stats"] = solver_stats

        energies_so_far.append(stats)
        out["energies_so_far"] = energies_so_far

        # save and error

        if (i % save_interval == 0) or i == (i0 + n_steps - 1):
            fname = f"{out_prefix}_{(i):04d}_rank_{nk.utils.mpi.rank}_of_{nk.utils.mpi.n_nodes}.pkl"
            with open(fname, "wb") as f:
                pickle.dump(out, f)
        if jnp.isnan(eloc).any():
            print("got nan\n Use a kernel which is positive definite and/or try to increase the regularization")
            break

        # prepare next step
        if args.uniq_weight_diag_reg:
            assert counts is not None
            diag_reg = actual_diag_reg / counts
            if args.dr_scale_n:
                diag_reg = diag_reg * counts.sum()
        x_train, y_train = x_train_new, y_train_new

if __name__ == "__main__":
    sys.exit(main(sys.argv))
