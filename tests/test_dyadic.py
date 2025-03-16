# Test for DyadicMemory
"""
The xcount variable below sets how many records would be stored & queried within memory
"""

import argparse

from time import time
import numpy as np
from sdm.mem_sdr import DyadicMemory as MemDyadicMemory  # noqa: F401
from sdm.sdr import DyadicMemory
from sdm.utils import random_sdrs


SDR_SIZE = 1000  # Size of SDRs in bits
SDR_BITS = 10  # Default number of ON bits (aka solidity)


def test_dyadic(mem, xcount):
    print(
        f"Testing DyadicMemory for {xcount} entries,\nsdr size is {SDR_SIZE} with {SDR_BITS} of ON bits"
    )
    # The number of records to write and query

    t = time()
    X = random_sdrs(xcount + 1, sdr_size=SDR_SIZE, on_bits=SDR_BITS)
    t = time() - t
    print(f"{xcount + 1} random SDRs generated in {int(t * 1000)}ms")

    t = time()
    for i, x in enumerate(X):
        if i == xcount:
            break
        y = X[i + 1]
        mem.store(
            x, y
        )  ######################################################## Store operation
    t = time() - t
    print(f"{xcount} writes in {int(t * 1000)} ms")

    print("Testing queries")

    size_errors = {}
    found = np.zeros((xcount, mem.P), dtype=np.uint16)
    t = time()

    for i in range(xcount):
        qresult = mem.query(
            X[i]
        )  ########################################## Query operations
        if qresult.size == mem.P:
            found[i] = qresult
        else:
            found[i] = X[i + 1]  #
            size_errors[i] = qresult

    t = time() - t

    print(f"{xcount} queries done in {int(t * 1000)}ms")

    print("Comparing results with expectations")
    if len(size_errors):
        print(f"{len(size_errors)} size errors, check size_errors dictionary")
    diff = X[1:] != found

    print(f"{(diff.sum(axis=1) > 0).sum()} differences check diff array")


parser = argparse.ArgumentParser(
    prog="dyadic tester", description="test the dyadic memory"
)
parser.add_argument("-c", "--count", type=int, default=45000)
parser.add_argument("-m", "--mem", action="store_true")
args = parser.parse_args()

if args.mem:
    mem = MemDyadicMemory(
        SDR_SIZE, SDR_BITS
    )  ######################################## Initialize the memory
else:
    mem = DyadicMemory(SDR_SIZE, SDR_BITS)

test_dyadic(mem, args.count)
