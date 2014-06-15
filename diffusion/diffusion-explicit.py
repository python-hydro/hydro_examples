# finite-difference implementation of the diffusion equation with first-order
# explicit time discretization
# 
# We are solving phi_t = k phi_xx
#
# We run at several resolutions and compute the error.  This uses a
# cell-centered finite-difference grid
#
# M. Zingale (2013-04-07)

import numpy
import pylab
import sys

class ccFDgrid:

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0):

        self.xmin = xmin
        self.xmax = xmax
        self.ng = ng
        self.nx = nx

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx

        # storage for the solution
        self.phi = numpy.zeros((nx+2*ng), dtype=numpy.float64)

    def scratchArray(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.nx+2*self.ng), dtype=numpy.float64)

    def fillBCs(self):
        """ fill the ghostcells with zero gradient (Neumann)
            boundary conditions """
        self.phi[0:self.ilo]  = self.phi[self.ilo]
        self.phi[self.ihi+1:] = self.phi[self.ihi]

    def phi_a(self, t, k, t0, phi1, phi2):
        """ analytic solution """

        xc = 0.5*(self.xmin + self.xmax)
        return (phi2 - phi1)*numpy.sqrt(t0/(t + t0)) * \
            numpy.exp(-0.25*(self.x-xc)**2/(k*(t + t0))) + phi1


    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if not len(e) == (2*self.ng + self.nx):
            return None

        return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))



def evolve(nx, k, C, tmax):

    ng = 1

    # create the grid
    g = ccFDgrid(nx, ng)

    # time info
    dt = C*0.5*g.dx**2/k
    t = 0.0

    # initialize the data
    g.phi[:] = g.phi_a(0.0, k, t0, phi1, phi2)

    # evolution loop
    phinew = g.scratchArray()
    
    while (t < tmax):

        # make sure we end right at tmax
        if (t + dt > tmax):
            dt = tmax - t

        # fill the boundary conditions
        g.fillBCs()

        alpha = k*dt/g.dx**2

        # loop over zones
        i = g.ilo
        while (i <= g.ihi):
        
            # explicit diffusion
            phinew[i] = g.phi[i] + alpha*(g.phi[i+1] - 2.0*g.phi[i] + g.phi[i-1])
                
            i += 1

        # store the updated solution
        g.phi[:] = phinew[:]

        t += dt

    return g



# diffusion coefficient
k = 1.0

# reference time
t0 = 1.e-4

# state coeffs
phi1 = 1.0
phi2 = 2.0


#-----------------------------------------------------------------------------
# solution at multiple times

# a characteristic timescale for diffusion if L^2/k
tmax = 0.0008

nx = 64

C = 0.8

ntimes = 4
tend = tmax/10.0**ntimes

c = ["0.5", "r", "g", "b", "k"]

while tend <= tmax:

    g = evolve(nx, k, C, tend)

    phi_analytic = g.phi_a(tend, k, t0, phi1, phi2)
    
    color = c.pop()
    pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], color=color, label="$t = %g$ s" % (tend))
    pylab.plot(g.x[g.ilo:g.ihi+1], phi_analytic[g.ilo:g.ihi+1], color=color, ls="--")

    tend = 10.0*tend


pylab.xlim(0.35,0.65)
#ax = pylab.gca()
#ax.set_yscale("log")

pylab.legend(frameon=False, fontsize="small")

pylab.xlabel("$x$")
pylab.ylabel(r"$\phi$")
pylab.title("explicit diffusion, nx = %d, C = %3.2f" % (nx, C))

pylab.savefig("diff-explicit-64.png")

#-----------------------------------------------------------------------------
# convergence

pylab.clf()

# a characteristic timescale for diffusion if L^2/k
tmax = 0.005


N = [16, 32, 64, 128, 256, 512]#, 1024]
ng = 1

# CFL number
C = 0.8


err = []

for nx in N:


    # compute the error
    g = evolve(nx, k, C, tmax)

    phi_analytic = g.phi_a(tmax, k, t0, phi1, phi2)

    err.append(g.norm(g.phi - phi_analytic))
    print g.dx, nx, err[-1]

    pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], label="N = %d" % (nx))


pylab.legend(frameon=False)
pylab.xlabel("$x$")
pylab.ylabel(r"$\phi$")
pylab.title("Explicit diffusion with varying resolution, C = %3.2f, t = %5.2g" % (C, tmax))

pylab.savefig("diffexplicit-res.png")



pylab.clf()

N = numpy.array(N, dtype=numpy.float64)
err = numpy.array(err)

pylab.scatter(N, err, color="r")
pylab.plot(N, err[len(N)-1]*(N[len(N)-1]/N)**2, color="k", label="$\mathcal{O}(\Delta x^2)$")

ax = pylab.gca()
ax.set_xscale('log')
ax.set_yscale('log')

pylab.xlabel(r"$N$")
pylab.ylabel(r"L2 norm of absolute error")
pylab.title("Convergence of Explicit Diffusion")

pylab.legend(frameon=False)

pylab.savefig("diffexplicit-converge.png")



#-----------------------------------------------------------------------------
# exceed the timestep limit

pylab.clf()

# a characteristic timescale for diffusion if L^2/k
tmax = 0.005

nx = 64

C = 2.0

g = evolve(nx, k, C, tmax)
phi_analytic = g.phi_a(tmax, k, t0, phi1, phi2)
    
pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], color="k")
pylab.plot(g.x[g.ilo:g.ihi+1], phi_analytic[g.ilo:g.ihi+1], color="k", ls="--")

pylab.xlim(0.35,0.65)
pylab.xlabel("$x$")
pylab.ylabel(r"$\phi$")
pylab.title("explicit diffusion, nx = %d, C = %3.2f, t = %5.2g" % (nx, C, tmax))

pylab.savefig("diff-explicit-64-bad.png")








    
