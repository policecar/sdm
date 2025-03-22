"""
This code is an implementation of the Triadic Memory and Dyadic Memory algorithms

Copyright (c) 2021-2022 Peter Overmann
Copyright (c) 2022 Cezar Totth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import numba


@numba.jit(nopython=True)
def xaddr(x, N):
    """
    Convert an SDR pattern to memory addresses using a dyadic encoding scheme.

    This function computes memory addresses based on pairs of active bits in the input pattern.
    Each pair (x[i], x[j]) where i > j produces an address based on a triangular indexing formula.

    Args:
        x (array-like):     A sorted SDR (indices of active bits).
        N (int):            The dimension of the full SDR vector.

    Returns:
        list: Memory addresses corresponding to all pairs of active bits in x.

    Note:
        The formula x[i] * (x[i] - 1) // 2 + x[j] maps each unique pair to
        a unique address in the triangular upper portion of an NxN matrix.
    """
    addr = []
    for i in range(1, len(x)):
        for j in range(i):
            addr.append(x[i] * (x[i] - 1) // 2 + x[j])
    return addr


@numba.jit(nopython=True)
def store_xy(mem, N, x, y):
    """
    Stores Y under key X
    Y and X have to be sorted sparsely encoded SDRs
    """
    for addr in xaddr(x, N):
        for j in y:
            mem[addr, j] = 1


@numba.jit(nopython=True)
def store_xyz(mem, x, y, z):
    """
    Stores X, Y, Z triplet in mem
    All X, Y, Z have to be sparse encoded SDRs
    """
    for ax in x:
        for ay in y:
            for az in z:
                mem[ax, ay, az] = 1


@numba.jit(nopython=True)
def query(mem, N, P, x):
    """
    Query in DyadicMemory
    """
    sums = np.zeros(mem.shape[1], dtype=np.uint32)
    for addr in xaddr(x, N):
        sums += mem[addr]
    return sums2sdr(sums, P)


@numba.jit(nopython=True)
def queryZ(mem, P, x, y):
    N = mem.shape[0]
    sums = np.zeros(N, dtype=np.uint32)
    for ax in x:
        for ay in y:
            sums += mem[ax, ay, :]
    return sums2sdr(sums, P)


@numba.jit(nopython=True)
def queryX(mem, P, y, z):
    N = mem.shape[0]
    sums = np.zeros(N, dtype=np.uint32)
    for ay in y:
        for az in z:
            sums += mem[:, ay, az]
    return sums2sdr(sums, P)


@numba.jit(nopython=True)
def queryY(mem, P, x, z):
    N = mem.shape[0]
    sums = np.zeros(N, dtype=np.uint32)
    for ax in x:
        for az in z:
            sums += mem[ax, :, az]
    return sums2sdr(sums, P)


@numba.jit(nopython=True)
def sums2sdr(sums, P):
    """
    Convert a vector of activation sums to an SDR with P active bits.

    This function selects the P highest values from the sums vector and
    returns their indices as a sparse representation.

    Args:
        sums (numpy.ndarray):   Vector of activation counts or values.
        P (int):                Number of active bits in the output sparse SDR.

    Returns:
        numpy.ndarray:          Indices of the P highest values in sums.
                                If there are fewer than P non-zero values,
                                returns indices of all non-zero values.

    Note:
        If multiple values tie for the P-th highest position,
        all indices with values >= the threshold will be included,
        potentially resulting in more than P active bits.
    """
    # this does what binarize() does in C
    ssums = sums.copy()
    ssums.sort()
    threshval = ssums[-P]
    if threshval == 0:
        return np.where(sums)[0]  # All non zero values
    else:
        return np.where(sums >= threshval)[0]  #


class TriadicMemory:
    """
    A class implementing a triadic sparse distributed memory
    that stores associations between three patterns (x, y, z).

    This memory structure can retrieve any one pattern when provided with the other two.
    """

    def __init__(self, N, P):
        """
        Initialize a triadic memory with dimensions N x N x N.
        """
        self.mem = np.zeros((N, N, N), dtype=np.uint8)
        self.P = P

    def store(self, x, y, z):
        """
        Store a triple association between patterns x, y, and z.

        Args:
           x, y, z (array-like):    Sorted sparse SDRs to associate.
        """
        store_xyz(self.mem, x, y, z)

    def query(self, x, y, z=None):
        """
        Query the memory for the missing pattern.
        Provide two patterns and set the third to None to retrieve it.

        Args:
           x, y, z: Two patterns as sorted sparse SDRs, with the third as None.

        Returns:
           The missing pattern as a sorted sparse SDR.
        """
        if z is None:
            return queryZ(self.mem, self.P, x, y)
        elif x is None:
            return queryX(self.mem, self.P, y, z)
        elif y is None:
            return queryY(self.mem, self.P, x, z)

    def query_X(self, y, z):
        """Query for x given y and z."""
        return queryX(self.mem, self.P, y, z)

    def query_Y(self, x, z):
        """Query for y given x and z."""
        return queryY(self.mem, self.P, x, z)

    def query_Z(self, x, y):
        """Query for z given x and y."""
        return queryZ(self.mem, self.P, x, y)

    def query_x_with_P(self, y, z, P):
        """
        Query for x with custom sparsity parameter P.

        Args:
           y, z: Patterns as sorted sparse SDRs.
           P (int): Custom number of active bits to use.
        """
        return queryX(self.mem, P, y, z)

    @property
    def N(self):
        """Return the dimension size of the memory."""
        return self.mem.shape[0]


class DyadicMemory:
    """
    An object-oriented interface for Sparse Distributed Memory (SDM)
    implementing a dyadic memory model.

    This class provides methods to store and retrieve Sparse Distributed Representations (SDRs)
    using an associative memory approach. It uses a dyadic addressing scheme
    where pairs of active bits in the input pattern determine memory addresses.

    """

    def __init__(self, N, P):
        """
        N is SDR vector size, e.g. 1000
        P is the count of solid bits, e.g. 10
        """
        self.mem = np.zeros((N * (N - 1) // 2, N), dtype=np.uint8)
        print(f"DyadicMemory size {self.mem.size / 1000000} M bytes")
        self.N = N
        self.P = P

    def store(self, x, y):
        """
        Store a pattern y associated with key x in memory
        by writing the pattern y to all memory locations addressed by x.

        Args:
            x (array-like):     The key pattern as a sorted sparse SDR.
            y (array-like):     The value pattern as a sorted sparse SDR.
        """
        store_xy(self.mem, self.N, x, y)

    def query(self, x):
        """
        Retrieve a pattern associated with key x from memory
        by reading from all memory locations addressed by x and constructing
        an output SDR based on activation counts.

        Args:
            x (array-like):     The key pattern as a sorted sparse SDR.

        Returns:
            array-like:         A sorted sparse SDR representing the retrieved pattern.
                                The output will have P active bits corresponding to the
                                highest activation counts.
        """
        return query(self.mem, self.N, self.P, x)
