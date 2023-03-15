# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a modified version of the netket statistics,
# which supports threading through tokens for mpi4jax
# which is necessary to avoid deadlocks on gpu

import jax.numpy as jnp

from netket.utils import mpi
from netket.utils.mpi import MPI_jax_comm


def subtract_mean(x, axis=None, *, token=None, comm=MPI_jax_comm):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        x: Input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.

    Returns:
        The resulting array.

    """
    # here we keep the dims, since automatic broadcasting of a scalar (shape () )
    # to an array produces errors when used inside of a function which is transposed
    # with jax.linear_transpose
    x_mean, token = mean(x, axis=axis, keepdims=True, token=token, comm=comm)
    return x - x_mean, token  # automatic broadcasting of x_mean


def mean(a, axis=None, keepdims: bool = False, *, token=None, comm=MPI_jax_comm):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. float64 intermediate and
    return values are used for integer inputs.

    Args:
        a: The input array
        axis: Axis or axes along which the means are computed. The default (None) is
            to compute the mean of the flattened array.
        keepdims: If True the output array will have the same number of dimensions as
            the input, with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis.

    """
    out = a.mean(axis=axis, keepdims=keepdims)

    return mpi.mpi_mean_jax(out, token=token, comm=comm)


def sum(a, axis=None, keepdims: bool = False, *, token=None, comm=MPI_jax_comm):
    """
    Compute the sum along the specified axis and over MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the mean is computed. The default (None) is to
              compute the mean of the flattened array.
        out: An optional pre-allocated array to fill with the result.
        keepdims: If True the output array will have the same number of dimensions as
              the input, with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis. If out is not none,
        returns out.

    """
    # if it's a numpy-like array...
    if hasattr(a, "shape"):
        # use jax
        a_sum = a.sum(axis=axis, keepdims=keepdims)
    else:
        # assume it's a scalar
        a_sum = jnp.asarray(a)

    return mpi.mpi_sum_jax(a_sum, token=token, comm=comm)


def var(a, axis=None, ddof: int = 0, *, token=None, comm=MPI_jax_comm):
    """
    Compute the variance mean along the specified axis and over MPI processes.
    Assumes same shape on all MPI processes.

    Args:
        a: The input array
        axis: Axis or axes along which the variance is computed. The default (None)
            is to compute the variance of the whole flattened array.
        out: An optional pre-allocated array to fill with the result.
        ddof: “Delta Degrees of Freedom”: the divisor used in the calculation
            is $N - ddof$, where $N$ represents the number of elements.
            By default ddof is zero.

    Returns:
        The array with reduced dimensions defined by axis. If out is not none,
        returns out.

    """
    m, token = mean(a, axis=axis, token=token, comm=comm)

    if axis is None:
        ssq = jnp.abs(a - m) ** 2.0
    else:
        ssq = jnp.abs(a - jnp.expand_dims(m, axis)) ** 2.0

    out, token = sum(ssq, axis=axis, token=token, comm=comm)

    n_all = total_size(a, axis=axis)
    out /= n_all - ddof

    return out, token


def total_size(a, axis=None):
    """
    Compute the total number of elements stored in the input array among all
    MPI processes.

    This function essentially returns MPI_sum_among_processes(a.size).

    Args:
        a: The input array.
        axis: If specified, only considers the total size of that axis.

    Returns:
        a.size or a.shape[axis], reduced among all MPI processes.
    """
    if axis is None:
        l_size = a.size
    else:
        l_size = a.shape[axis]

    # TODO: This function cannot call Python MPI because if it gets called on shape
    # inference when compiling. Therefore if only one mpi rank is compiling this
    # leads to deadlocks.
    # We should refactor all this logic.
    # return mpi_sum(l_size)
    return l_size * mpi.n_nodes