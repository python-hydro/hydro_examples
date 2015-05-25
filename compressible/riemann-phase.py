# plot the Hugoniot loci for a compressible Riemann problem

import numpy as np
import pylab

gamma = 1.4

class State:
    """ a simple container """
    def __init__(self, p=1.0, u=0.0, rho=1.0):
         self.p = p
         self.u = u
         self.rho = rho


def u_hugoniot(p, state, dir):

    c = np.sqrt(gamma*state.p/state.rho)

    u = np.zeros_like(p)

    if dir == "left":
        s = 1.0
    elif dir == "right":
        s = -1.0

    # rarefaction
    index_rare = np.where(p < state.p)
    u[index_rare] = state.u + s*(2.0*c/(gamma-1.0))* \
        (1.0 - (p[index_rare]/state.p)**((gamma-1.0)/(2.0*gamma)))

    # shock
    index_shock = np.where(p >= state.p)
    beta = (gamma+1.0)/(gamma-1.0)
    u[index_shock] = state.u + s*(2.0*c/np.sqrt(2.0*gamma*(gamma-1.0)))* \
        (1.0 - p[index_shock]/state.p)/np.sqrt(1.0 + beta*p[index_shock]/state.p)

    return u, index_shock, index_rare



# setup the problem -- Sod
left = State(p = 1.0, u = 0.0, rho = 1.0)
right = State(p = 0.1, u = 0.0, rho = 0.125)

# make the plots
N = 200

p_min = 0.0
p_max = 1.5

p = np.arange(N)*(p_max - p_min)/N + p_min


# curves
u_left, ish, ir = u_hugoniot(p, left, "left")

pylab.plot(p[ish], u_left[ish], c="b", ls=":", lw=2)
pylab.plot(p[ir], u_left[ir], c="b", ls="-", lw=2)
pylab.scatter([left.p], [left.u], marker="x", c="b", s=40)


u_right, ish, ir = u_hugoniot(p, right, "right")

pylab.plot(p[ish], u_right[ish], c="r", ls=":", lw=2)
pylab.plot(p[ir], u_right[ir], c="r", ls="-", lw=2)
pylab.scatter([right.p], [right.u], marker="x", c="r", s=40)


du = 0.025*(max(np.max(u_left), np.max(u_right)) - 
           min(np.min(u_left), np.min(u_right)))

pylab.text(left.p, left.u+du, "left", 
           horizontalalignment="center", color="b")

pylab.text(right.p, right.u+du, "right", 
           horizontalalignment="center", color="r")


pylab.xlim(p_min, p_max)

pylab.xlabel(r"$p$", fontsize="large")
pylab.ylabel(r"$u$", fontsize="large")

legs = []
legnames = []

legs.append(pylab.Line2D((0,1),(0,0), color="k", ls=":", marker=None))
legnames.append("shock")

legs.append(pylab.Line2D((0,1),(0,0), color="k", ls="-", marker=None))
legnames.append("rarefaction")

pylab.legend(legs, legnames, frameon=False, loc="best")

pylab.tight_layout()

pylab.savefig("riemann-phase.png")
pylab.savefig("riemann-phase.eps")
