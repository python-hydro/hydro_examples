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


class Grid1d:

    def __init__(self, nx, ng=1, xmin=0.0, xmax=1.0):
        """ grid class initialization """
        
        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin)/nx
        self.x = (numpy.arange(nx+2*ng) + 0.5 - ng)*self.dx + xmin

        self.ilo = ng
        self.ihi = ng+nx-1

        # storage for the solution
        self.phi = numpy.zeros((nx+2*ng), dtype=numpy.float64)

    def fillBC(self):
        """ fill the Neumann BCs """

        # Neumann BCs
        self.phi[0:self.ilo]  = self.phi[self.ilo]
        self.phi[self.ihi+1:] = self.phi[self.ihi]


    def scratch_array(self):
        return numpy.zeros((2*self.ng+self.nx), dtype=numpy.float64)


    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if not len(e) == (2*self.ng + self.nx):
            return None

        return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))



def phi_a(t, k, x, xc, t0, phi1, phi2):
    """ analytic solution for the diffusion of a Gaussian """

    return (phi2 - phi1)*numpy.sqrt(t0/(t + t0)) * \
        numpy.exp(-0.25*(x-xc)**2/(k*(t + t0))) + phi1


class Simulation:

    def __init__(self, grid, k=1.0):
        self.grid = grid
        self.t = 0.0
        self.k = k  # diffusion coefficient


    def init_cond(self, name, *args):

        if name == "gaussian":

            # initialize the data
            xc = 0.5*(self.grid.xmin + self.grid.xmax)
            t0, phi1, phi2 = args
            self.grid.phi[:] = phi_a(0.0, self.k, self.grid.x, xc, t0, phi1, phi2)


    def evolve(self, C, tmax):

        gr = self.grid

        # time info
        dt = C*0.5*gr.dx**2/self.k


        phinew = gr.scratch_array()
    
        while self.t < tmax:

            # make sure we end right at tmax
            if self.t + dt > tmax:
                dt = tmax - self.t

            # fill the boundary conditions
            gr.fillBC()

            alpha = self.k*dt/gr.dx**2

            # loop over zones
            i = g.ilo
            while (i <= g.ihi):
        
                # explicit diffusion
                phinew[i] = gr.phi[i] + \
                            alpha*(gr.phi[i+1] - 2.0*gr.phi[i] + gr.phi[i-1])
                
                i += 1

            # store the updated solution
            gr.phi[:] = phinew[:]

            self.t += dt



#-----------------------------------------------------------------------------
# diffusion coefficient
k = 1.0

# reference time
t0 = 1.e-4

# state coeffs
phi1 = 1.0
phi2 = 2.0


# solution at multiple times

# a characteristic timescale for diffusion if L^2/k
tmax = 0.0008

nx = 128

C = 0.8

ntimes = 4
tend = tmax/10.0**ntimes

c = ["0.5", "r", "g", "b", "k"]

while tend <= tmax:

    g = Grid1d(nx, ng=2)
    s = Simulation(g, k=k)
    s.init_cond("gaussian", t0, phi1, phi2)
    s.evolve(C, tend)

    xc = 0.5*(g.xmin + g.xmax)
    phi_analytic = phi_a(tend, k, g.x, xc, t0, phi1, phi2)
    
    color = c.pop()
    pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], 
               "x", color=color, label="$t = %g$ s" % (tend))
    pylab.plot(g.x[g.ilo:g.ihi+1], phi_analytic[g.ilo:g.ihi+1], 
               color=color, ls=":")

    tend = 10.0*tend


pylab.xlim(0.35,0.65)

pylab.legend(frameon=False, fontsize="small")

pylab.xlabel("$x$")
pylab.ylabel(r"$\phi$")
pylab.title("explicit diffusion, nx = %d, C = %3.2f" % (nx, C))

pylab.savefig("diff-explicit-{}.png".format(nx))

#-----------------------------------------------------------------------------
# convergence

# a characteristic timescale for diffusion is L^2/k
tmax = 0.005

t0 = 1.e-4
phi1 = 1.0
phi2 = 2.0

k = 1.0

N = [32, 64, 128, 256, 512]


# CFL number
C = 0.8

err = []

for nx in N:

    print nx

    # the present C-N discretization
    g = Grid1d(nx, ng=1)
    s = Simulation(g, k=k)
    s.init_cond("gaussian", t0, phi1, phi2)
    s.evolve(C, tmax)
    
    xc = 0.5*(g.xmin + g.xmax)
    phi_analytic = phi_a(tmax, k, g.x, xc, t0, phi1, phi2)

    err.append(g.norm(g.phi - phi_analytic))


pylab.clf()

N = numpy.array(N, dtype=numpy.float64)
err = numpy.array(err)

pylab.scatter(N, err, color="r", label="explicit diffusion")
pylab.loglog(N, err[len(N)-1]*(N[len(N)-1]/N)**2, color="k", label="$\mathcal{O}(\Delta x^2)$")


pylab.xlabel(r"$N$")
pylab.ylabel(r"L2 norm of absolute error")
pylab.title("Convergence of Explicit Diffusion, C = %3.2f, t = %5.2g" % (C, tmax))

pylab.ylim(1.e-6, 1.e-2)
pylab.legend(frameon=False, fontsize="small")

pylab.savefig("diffexplicit-converge-{}.png".format(C))



#-----------------------------------------------------------------------------
# exceed the timestep limit

pylab.clf()

# a characteristic timescale for diffusion if L^2/k
tmax = 0.005

nx = 64

C = 2.0

g = Grid1d(nx, ng=2)
s = Simulation(g, k=k)
s.init_cond("gaussian", t0, phi1, phi2)
s.evolve(C, tend)

xc = 0.5*(g.xmin + g.xmax)
phi_analytic = phi_a(tend, k, g.x, xc, t0, phi1, phi2)
    
pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], 
           "x-", color="r", label="$t = %g$ s" % (tend))
pylab.plot(g.x[g.ilo:g.ihi+1], phi_analytic[g.ilo:g.ihi+1], 
           color="0.5", ls=":")

pylab.xlim(0.35,0.65)
pylab.xlabel("$x$")
pylab.ylabel(r"$\phi$")
pylab.title("explicit diffusion, nx = %d, C = %3.2f, t = %5.2g" % (nx, C, tmax))

pylab.savefig("diff-explicit-64-bad.png")







    
