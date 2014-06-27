"""
2nd-order accurate finite-volume implementation of linear advection with 
piecewise linear slope reconstruction.

We are solving a_t + u a_x = 0

This script defines two classes:

 -- the Grid1d class that manages a cell-centered grid and holds the 
    data that lives on that grid

 -- the Simulation class that is built on a Grid1d object and defines
    everything needed to do a advection.

Options for several different slope limiters are provided.

M. Zingale

"""

import numpy
import pylab
import math


# helper functions for the limiting
def minmod(a, b):
    if (abs(a) < abs(b) and a*b > 0.0):
        return a
    elif (abs(b) < abs(a) and a*b > 0.0):
        return b
    else:
        return 0.0

def maxmod(a, b):
    if (abs(a) > abs(b) and a*b > 0.0):
        return a
    elif (abs(b) > abs(a) and a*b > 0.0):
        return b
    else:
        return 0.0


class Grid1d:

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0):

        self.ng = ng
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx

        # storage for the solution
        self.a = numpy.zeros((nx+2*ng), dtype=numpy.float64)


    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.nx+2*self.ng), dtype=numpy.float64)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        n = 0
        while n < self.ng:
            # left boundary
            self.a[self.ilo-1-n] = self.a[self.ihi-n]

            # right boundary
            self.a[self.ihi+1+n] = self.a[self.ilo+n]
            n += 1


    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if not len(e) == (2*self.ng + self.nx):
            return None

        return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))


class Simulation:

    def __init__(self, grid, u, C=0.8, slope_type="centered"):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.u = u   # the constant advective velocity
        self.C = C   # CFL number
        self.slope_type = slope_type


    def init_cond(self, type="tophat"):
        """ initialize the data """
        if type == "tophat":
            self.grid.a[:] = 0.0
            self.grid.a[numpy.logical_and(self.grid.x >= 0.333, 
                                          self.grid.x <= 0.666)] = 1.0

        elif type == "sine":
            self.grid.a[:] = numpy.sin(2.0*math.pi*self.grid.x/(self.grid.xmax-self.grid.xmin))

        elif type == "gaussian":
            self.grid.a[:] = 1.0 + numpy.exp(-60.0*(self.grid.x - 0.5)**2)


    def timestep(self):
        """ return the advective timestep """
        return self.C*self.grid.dx/self.u


    def period(self):
        """ return the period for advection with velocity u """
        return (self.grid.xmax - self.grid.xmin)/self.u


    def states(self, dt):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes
        g = self.grid
        slope = g.scratch_array()

        g = self.grid
        
        if self.slope_type == "godunov":

            # piecewise constant = 0 slopes
            slope[:] = 0.0

        elif self.slope_type == "centered":

            # unlimited centered difference slopes
            i = g.ilo-1
            while i <= g.ihi+1:
                slope[i] = 0.5*(g.a[i+1] - g.a[i-1])/g.dx
                i += 1

        elif self.slope_type == "minmod":

            # minmod limited slope
            i = g.ilo-1
            while i <= g.ihi+1:
                slope[i] = minmod( (g.a[i] - g.a[i-1])/g.dx, 
                                   (g.a[i+1] - g.a[i])/g.dx )
                i += 1
        
        elif self.slope_type == "MC":

            # MC limiter
            i = g.ilo-1
            while i <= g.ihi+1:
                slope[i] = minmod(minmod( 2.0*(g.a[i] - g.a[i-1])/g.dx, 
                                          2.0*(g.a[i+1] - g.a[i])/g.dx ),
                                  0.5*(g.a[i+1] - g.a[i-1])/g.dx)
                i += 1

        elif self.slope_type == "superbee":

            # superbee limiter
            i = g.ilo-1
            while i <= g.ihi+1:
                A = minmod( (g.a[i+1] - g.a[i])/g.dx,
                            2.0*(g.a[i] - g.a[i-1])/g.dx )

                B = minmod( (g.a[i] - g.a[i-1])/g.dx,
                            2.0*(g.a[i+1] - g.a[i])/g.dx )
            
                slope[i] = maxmod(A, B)
                i += 1



        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that thre are 1 more interfaces
        # than zones
        al = g.scratch_array()
        ar = g.scratch_array()

        i = g.ilo
        while i <= g.ihi+1:

            # left state on the current interface comes from zone i-1
            al[i] = g.a[i-1] + 0.5*g.dx*(1.0 - u*dt/g.dx)*slope[i-1]

            # right state on the current interface comes from zone i
            ar[i] = g.a[i] - 0.5*g.dx*(1.0 + u*dt/g.dx)*slope[i]

            i += 1

        return al, ar


    def riemann(self, al, ar):
        """ 
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux 
        """

        if self.u > 0.0:
            return self.u*al
        else:
            return self.u*ar


    def update(self, dt, flux):
        """ conservative update """

        g = self.grid

        anew = g.scratch_array()

        anew[g.ilo:g.ihi+1] = g.a[g.ilo:g.ihi+1] + \
            dt/g.dx * (flux[g.ilo:g.ihi+1] - flux[g.ilo+1:g.ihi+2])

        return anew


    def evolve(self, num_periods=1):
        """ evolve the linear advection equation """
        self.t = 0.0
        g = self.grid

        tmax = num_periods*self.period()


        # main evolution loop
        while (self.t < tmax):

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if (self.t + dt > tmax):
                dt = tmax - self.t

            # get the interface states
            al, ar = self.states(dt)

            # solve the Riemann problem at all interfaces
            flux = self.riemann(al, ar)
        
            # do the conservative update
            anew = self.update(dt, flux)

            g.a[:] = anew[:]

            self.t += dt


if __name__ == "__main__":

    #-------------------------------------------------------------------------
    # convergence test
    problem = "gaussian"

    xmin = 0.0
    xmax = 1.0
    ng = 2
    N = [32, 64, 128, 256]

    err = []

    for nx in N:

        g = Grid1d(nx, ng, xmin=xmin, xmax=xmax)

        u = 1.0
        s = Simulation(g, u, C=0.8, slope_type="centered")
        s.init_cond("gaussian")
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        # compute the error
        err.append(g.norm(g.a - ainit))
        print g.dx, nx, err[-1]


    pylab.clf()

    N = numpy.array(N, dtype=numpy.float64)
    err = numpy.array(err)

    pylab.scatter(N, err, color="r")
    pylab.plot(N, err[len(N)-1]*(N[len(N)-1]/N)**2, 
               color="k", label="2nd order convergence")

    ax = pylab.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    pylab.xlabel("N")
    pylab.ylabel("absolute error")

    pylab.legend(frameon=False, fontsize="small")
    
    pylab.savefig("plm-converge.png")


    #-----------------------------------------------------------------------------
    # different limiters: run both the Gaussian and tophat

    xmin = 0.0
    xmax = 1.0
    nx = 128
    ng = 2

    u = 1.0

    g= Grid1d(nx, ng, xmin=xmin, xmax=xmax)

    for p in ["gaussian", "tophat"]:
        pylab.clf()

        s = Simulation(g, u, C=0.8, slope_type="godunov")
        s.init_cond(p)
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        pylab.subplot(231)

        pylab.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1], color="r")
        pylab.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1], ls=":", color="0.5")

        pylab.title("piecewise constant")


        s = Simulation(g, u, C=0.8, slope_type="centered")
        s.init_cond(p)
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        pylab.subplot(232)
        
        pylab.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1], color="r")
        pylab.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1], ls=":", color="0.5")

        pylab.title("centered (unlimited)")


        s = Simulation(g, u, C=0.8, slope_type="minmod")
        s.init_cond(p)
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        pylab.subplot(233)

        pylab.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1], color="r")
        pylab.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1], ls=":", color="0.5")
        
        pylab.title("minmod limiter")


        s = Simulation(g, u, C=0.8, slope_type="MC")
        s.init_cond(p)
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        pylab.subplot(234)

        pylab.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1], color="r")
        pylab.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1], ls=":", color="0.5")

        pylab.title("MC limiter")


        s = Simulation(g, u, C=0.8, slope_type="superbee")
        s.init_cond(p)
        ainit = s.grid.a.copy()

        s.evolve(num_periods=5)

        pylab.subplot(235)

        pylab.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1], color="r")
        pylab.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1], ls=":", color="0.5")

        pylab.title("superbee limiter")


        f = pylab.gcf()
        f.set_size_inches(10.0,7.0)

        pylab.tight_layout()
        
        pylab.savefig("fv-{}-limiters.png".format(p), bbox_inches="tight")

