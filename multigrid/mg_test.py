#!/usr/bin/env python

"""

an example of using the multigrid class to solve Laplace's equation.  Here, we
solve

u_xx = sin(x)
u = 0 on the boundary [0,1]

The analytic solution is u(x) = -sin(x) + x sin(1)

"""
#from io import *
import numpy
import pylab

import multigrid

# the analytic solution
def true(x):
    return -numpy.sin(x) + x*numpy.sin(1.0)


# the L2 error norm
def error(myg, r):

    # L2 norm of elements in r, multiplied by dx to
    # normalize
    return numpy.sqrt(myg.dx*numpy.sum((r[myg.ilo:myg.ihi+1]**2)))


# the righthand side
def f(x):
    return numpy.sin(x)

                
# test the multigrid solver
nx = 256


# create the multigrid object
a = multigrid.CellCenterMG1d(nx,
                             xl_BC_type="dirichlet", xr_BC_type="dirichlet",
                             verbose=1, true_function=true)

# initialize the solution to 0
a.init_zeros()

# initialize the RHS using the function f
a.init_RHS(f(a.x))

# solve to a relative tolerance of 1.e-11
elist, rlist = a.solve(rtol=1.e-11)

Ncycle = numpy.arange(len(elist)) + 1


# get the solution 
v = a.get_solution()

# compute the error from the analytic solution
e = v - true(a.x)

print " L2 error from true solution = %g\n rel. err from previous cycle = %g\n num. cycles = %d" % \
      (error(a.soln_grid, e), a.relative_error, a.num_cycles)



pylab.clf()

pylab.plot(a.x[a.ilo:a.ihi+1], true(a.x[a.ilo:a.ihi+1]), color="r")
pylab.xlabel("x")
pylab.ylabel("$\phi$")

pylab.ylim([1.1*min(true(a.x[a.ilo:a.ihi+1])),0.0])
f = pylab.gcf()
f.set_size_inches(10.0,4.5)


pylab.savefig("phi_analytic.png")


pylab.clf()

pylab.plot(Ncycle, numpy.array(elist), color="k", label=r"$||e||$")
pylab.plot(Ncycle, numpy.array(rlist), "--", color="k", label=r"$||r||$")

pylab.xlabel("# of V-cycles")
pylab.ylabel("L2 norm of error")

ax = pylab.gca()
ax.set_yscale('log')

f = pylab.gcf()

f.set_size_inches(8.0,6.0)

pylab.legend(frameon=False)

pylab.tight_layout()

pylab.savefig("mg_error_vs_cycle.png")
pylab.savefig("mg_error_vs_cycle.eps")


