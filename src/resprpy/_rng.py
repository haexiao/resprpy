"""R's Mersenne-Twister RNG engine (R 4.x), ported bit-exactly.

Key facts from R-4.5.3/src/main/RNG.c:
- `typedef unsigned int Int32` in Random.h -- ALL state and arithmetic is
  UNSIGNED 32-bit (logical shifts).
- RNG_Init: seed is scrambled with 50 LCG steps, then 625 LCG values fill
  i_seed[0..624]; FixupSeeds sets i_seed[0] (= mti) to 624.
- MT_genrand: standard twist with mag01; tempering shifts are logical.
- unif_rand: fixup((double)y * 2^-32) with y unsigned.
- R_unif_index: rejection sampling of floor(unif * p).
"""

import math

import numpy as np

N = 624
M = 397
MATRIX_A = np.uint32(0x9908B0DF)
UPPER_MASK = np.uint32(0x80000000)
LOWER_MASK = np.uint32(0x7FFFFFFF)
MASK_B = np.uint32(0x9D2C5680)
MASK_C = np.uint32(0xEFC60000)
U32 = np.uint32


def _u32(v):
    """Wrap a Python int into unsigned 32-bit."""
    return np.uint32(v & 0xFFFFFFFF)


class RMersenneTwister:
    """R's MT19937 (RNG_Init path). Seed -> 50 LCG steps -> 625 LCG fills."""

    def __init__(self, seed):
        s = int(seed) & 0xFFFFFFFF
        # RNG_Init: initial scrambling (50 LCG steps), unsigned wrap
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        # MERSENNE_TWISTER branch: 625 i_seed values via LCG
        i_seed = []
        for _ in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            i_seed.append(_u32(s))
        # FixupSeeds(initial=1): I1 (= dummy[0] = mti) <- 624
        self._mti = 624
        self._mt = np.array(i_seed[1:625], dtype=np.uint32)

    def _genrand(self):
        """R MT_genrand(): twist + tempering, returns unsigned y."""
        mt = self._mt
        mti = self._mti
        if mti >= N:
            kk = 0
            for kk in range(N - M):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + M] ^ (y >> U32(1)) ^ (MATRIX_A if (y & U32(1)) else U32(0))
            for kk in range(N - M, N - 1):
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
                mt[kk] = mt[kk + (M - N)] ^ (y >> U32(1)) ^ (MATRIX_A if (y & U32(1)) else U32(0))
            y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)
            mt[N - 1] = mt[M - 1] ^ (y >> U32(1)) ^ (MATRIX_A if (y & U32(1)) else U32(0))
            mti = 0
        y = mt[mti]
        mti += 1
        y ^= y >> U32(11)
        y ^= (y << U32(7)) & MASK_B
        y ^= (y << U32(15)) & MASK_C
        y ^= y >> U32(18)
        self._mti = mti
        return int(y)

    @staticmethod
    def _fixup(x):
        """R fixup(): ensure 0 and 1 are never returned."""
        i2_32m1 = 2.3283064365386963e-10
        if x <= 0.0:
            return 0.5 * i2_32m1
        if (1.0 - x) <= 0.0:
            return 1.0 - 0.5 * i2_32m1
        return x

    def unif(self):
        """R unif_rand(): fixup((double)y * 2^-32), y unsigned."""
        y = self._genrand()
        return self._fixup(float(y) * 2.3283064365386963e-10)

    def runif(self, n, min_=0.0, max_=1.0):
        return [min_ + (max_ - min_) * self.unif() for _ in range(n)]

    def rbits(self, bits):
        """R rbits(): random int < 2^bits, built from 16-bit chunks of unif."""
        v = 0
        n = 0
        while n <= bits:
            v1 = int(self.unif() * 65536.0)  # floor(unif_rand() * 65536)
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def r_unif_index(self, dn):
        """R R_unif_index() (REJECTION kind, default): rejection sample."""
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self.rbits(bits)
            if not (dn <= dv):
                return dv

    def sample_int(self, n, size, replace=True):
        """R sample.int(n, size, replace=TRUE): 1-based results."""
        out = []
        for _ in range(size):
            out.append(self.r_unif_index(n) + 1)
        return out

    def sample(self, x, size, replace=True):
        """R sample(x, size, replace=TRUE) for integer x (1-based vector)."""
        if isinstance(x, int):
            n = x
        else:
            x = list(x)
            n = len(x)
        idx = self.sample_int(n, size, replace=replace)
        if isinstance(x, int):
            return idx
        return [x[i - 1] for i in idx]
