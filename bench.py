"""bench.py - speed benchmark: resprpy vs respR (same data from CSV)."""
import time

import numpy as np
import resprpy as p

BASE = "tests/reference/"


def bench(label, fn, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"{label} {best:.4f}")


u = np.genfromtxt(BASE + "urchins.csv", delimiter=",", skip_header=1)
d2k = np.genfromtxt("reference/bench_2k.csv", delimiter=",", skip_header=1)
d10k = np.genfromtxt("reference/bench_10k.csv", delimiter=",", skip_header=1)

bench("calc_rate_271", lambda: p.calc_rate(u[:, :2], from_=0, to=45, by="time"))
bench("calc_rate_2k", lambda: p.calc_rate(d2k, from_=0, to=1800, by="time"))
bench("calc_rate_10k", lambda: p.calc_rate(d10k, from_=0, to=9000, by="time"))
bench("auto_rate_271", lambda: p.auto_rate(u[:, :2]))
bench("auto_rate_2k", lambda: p.auto_rate(d2k))
bench("auto_rate_10k", lambda: p.auto_rate(d10k))
