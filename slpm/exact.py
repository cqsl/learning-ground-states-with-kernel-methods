import numpy as np


def ising1d_energy(L, h):
    """
    analytically compute the ground state energy of the 1D TFI model
    with nearest-neightbour interactions and periodic bounary conditions

    L: length of the chain
    h: strength of the transverse field
    """

    def _Epsilon(k, h):
        eps = 1 + h**2 + 2 * h * np.cos(k)
        return 2 * np.sqrt(eps)

    i = np.arange(L)
    k = np.pi * (2 * i + 1) / L
    energy = _Epsilon(k, h).sum()
    return -0.5 * energy


# works for even L
def ising1d_gap(L, h, return_energies=False, odd=False):
    """
    analytically compute the gap of the 1D TFI model
    for even chains with periodic boundary conditions

    L: length of the chain
    h: strength of the transverse field

    if odd==True:
        returns the gap in the odd parity sector
    else (default):
        returns the gap in the even sector (where the ground state lies)

    if return_energies=True:
        returns E0_even, E0_odd, E1_even, E1_odd
    """

    def _Epsilon(k, h):
        eps = 1 + h**2 + 2 * h * np.cos(2 * np.pi * k / L)
        return 2 * np.sqrt(eps)

    assert L % 2 == 0  # chain needs to be even

    k_even = np.arange(-(L - 1) / 2, (L - 1) / 2 + 0.00001, 1)
    k_odd = np.arange(-L / 2, L / 2 - 1 + 0.00001, 1)

    E0_even = -0.5 * _Epsilon(k_even, h).sum()
    E0_odd = -0.5 * _Epsilon(k_odd, h).sum()
    if h > 1:
        E0_odd += +2 * h - 2

    E1_even = E0_even + _Epsilon(-(L - 1) / 2, h) + _Epsilon((L - 1) / 2, h)
    E1_odd = -0.5 * _Epsilon(k_odd[k_odd != 0][1:], h).sum() + _Epsilon(-L / 2 + 1, h) - 2 * h

    gap_even = E1_even - E0_even
    gap_odd = E1_odd - E0_odd
    if return_energies:
        assert not odd
        return E0_even, E0_odd, E1_even, E1_odd
    elif odd:
        return float(gap_odd)
    else:
        return float(gap_even)
