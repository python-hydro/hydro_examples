"""
Comparing efficiency (accuracy vs runtime and memory) for various schemes.

Note: needs 'pip install memory_profiler' first.

At least for now, the memory usage seems to be completely misleading.
"""

import timeit
# from memory_profiler import memory_usage
import numpy
from matplotlib import pyplot
import advection
import weno
import dg


def run_weno(order=2, nx=32):
    g = advection.Grid1d(nx, ng=order+1, xmin=0, xmax=1)
    s = weno.WENOSimulation(g, 1, C=0.5, weno_order=order)
    s.init_cond("sine")
    s.evolve_scipy()


def run_dg(m=3, nx=8, limiter=None):
    g = dg.Grid1d(nx, ng=1, xmin=0, xmax=1, m=m)
    s = dg.Simulation(g, 1, C=0.5/(2*m+1), limiter=limiter)
    s.init_cond("sine")
    s.evolve_scipy()


def weno_string(order=2, nx=32):
    return f"run_weno(order={order}, nx={nx})"


def dg_string(m=3, nx=8, limiter=None):
    return f"run_dg(m={m}, nx={nx}, limiter='{limiter}')"


def time_weno(order=2, nx=32):
    return timeit.timeit(weno_string(order, nx),
                         globals=globals(), number=1)


def time_dg(m=3, nx=8, limiter=None):
    return timeit.timeit(dg_string(m, nx, limiter),
                         globals=globals(), number=1)


def errs_weno(order=2, nx=32):
    g = advection.Grid1d(nx, ng=order+1, xmin=0, xmax=1)
    s = weno.WENOSimulation(g, 1, C=0.5, weno_order=order)
    s.init_cond("sine")
    a_init = g.a.copy()
    s.evolve_scipy()
    return g.norm(g.a - a_init)


def errs_dg(m=3, nx=8, limiter=None):
    g = dg.Grid1d(nx, ng=1, xmin=0, xmax=1, m=m)
    s = dg.Simulation(g, 1, C=0.5/(2*m+1), limiter=limiter)
    s.init_cond("sine")
    a_init = g.a.copy()
    s.evolve_scipy_jit()
    return g.norm(g.a - a_init)


weno_orders = [3, 5, 7]
weno_N = [12, 16, 24, 32, 54, 64, 96, 128]
# weno_orders = [3]
# weno_N = [24, 32, 54]
weno_times = numpy.zeros((len(weno_orders), len(weno_N)))
weno_errs = numpy.zeros_like(weno_times)
# weno_memory = numpy.zeros_like(weno_times)
# weno_opt_memory = numpy.zeros_like(weno_opt_times)

# Do one evolution to kick-start numba
run_weno(3, 8)
run_dg(3, 8)

for i_o, order in enumerate(weno_orders):
    for i_n, nx in enumerate(weno_N):
        print(f"WENO{order}, {nx} points")
        weno_errs[i_o, i_n] = errs_weno(order, nx)
        weno_times[i_o, i_n] = time_weno(order, nx)
#        weno_memory[i_o, i_n] = max(memory_usage((run_weno, (order, nx))))

dg_ms = [2, 4, 8]
dg_N = 2**numpy.array(range(3, 7))
# dg_ms = [2]
# dg_N = 2**numpy.array(range(3, 6))
dg_moment_times = numpy.zeros((len(dg_ms), len(dg_N)))
dg_moment_errs = numpy.zeros_like(dg_moment_times)
# dg_nolimit_memory = numpy.zeros_like(dg_nolimit_times)
# dg_moment_memory = numpy.zeros_like(dg_moment_times)

for i_m, m in enumerate(dg_ms):
    for i_n, nx in enumerate(dg_N):
        print(f"DG, m={m}, {nx} points")
        dg_moment_errs[i_m, i_n] = errs_dg(m, nx, 'moment')
        dg_moment_times[i_m, i_n] = time_dg(m, nx, 'moment')
#        dg_moment_memory[i_m, i_n] = max(memory_usage((run_dg,
#                                                      (m, nx, 'moment'))))

colors = "brk"
fig, ax = pyplot.subplots(1, 1)
for i_o, order in enumerate(weno_orders):
    ax.loglog(weno_times[i_o, :], weno_errs[i_o, :], f'{colors[i_o]}o-',
              label=f'WENO, r={order}')
colors = "gyc"
for i_m, m in enumerate(dg_ms):
    ax.loglog(dg_moment_times[i_m, :], dg_moment_errs[i_m, :],
              f'{colors[i_m]}^:', label=rf'DG, $m={m}$')
ax.set_xlabel("Runtime [s]")
ax.set_ylabel(r"$\|$Error$\|_2$")
ax.set_title("Efficiency of WENO vs DG")
fig.tight_layout()
lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.savefig('dg_weno_efficiency.pdf',
            bbox_extra_artists=(lgd,), bbox_inches='tight')
pyplot.show()
