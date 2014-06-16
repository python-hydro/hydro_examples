# 2nd-order accurate finite-volume implementation of the inviscid Burger's 
# equation with piecewise linear slope reconstruction
# 
# We are solving u_t + u u_x = 0 with outflow boundary conditions
#
# M. Zingale (2013-03-26)

import numpy
import pylab
import math
import sys

def minmod(a, b):
    if (abs(a) < abs(b) and a*b > 0.0):
        return a
    elif (abs(b) < abs(a) and a*b > 0.0):
        return b
    else:
        return 0.0


class Grid1d:

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0):

        self.xmin = xmin
        self.xmax = xmax
        self.ng = ng
        self.nx = nx

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx
        self.xl = xmin + (numpy.arange(nx+2*ng)-ng)*self.dx
        self.xr = xmin + (numpy.arange(nx+2*ng)-ng+1.0)*self.dx

        # storage for the solution
        self.u = numpy.zeros((nx+2*ng), dtype=numpy.float64)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.nx+2*self.ng), dtype=numpy.float64)

    def fillBCs(self):
        """ fill all ghostcells with outflow """

        # left boundary
        self.u[0:self.ilo] = self.u[self.ilo]

        # right boundary
        self.u[self.ihi+1:] = self.u[self.ihi]


class Simulation:

    def __init__(self, grid):
        self.grid = grid
        self.t = 0.0

        self.uinit = None
    

    def initCond(self, type="tophat"):

        if type == "tophat":
            self.grid.u[numpy.logical_and(self.grid.x >= 0.333, 
                                          self.grid.x <= 0.666)] = 1.0

        elif type == "sine":
            self.grid.u[:] = 1.0

            index = numpy.logical_and(self.grid.x >= 0.333, 
                                      self.grid.x <= 0.666)
            self.grid.u[index] += \
                0.5*numpy.sin(2.0*math.pi*(self.grid.x[index]-0.333)/0.333)

        elif type == "rarefaction":
            self.grid.u[:] = 1.0 
            self.grid.u[self.grid.x > 0.5] = 2.0

        self.uinit = self.grid.u.copy()


    def timestep(self, C):
        return C*self.grid.dx/max(abs(self.grid.u[self.grid.ilo:
                                                  self.grid.ihi+1]))


    def states(self, dt):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes -- 2nd order MC limiter
        ib = self.grid.ilo-1
        ie = self.grid.ihi+1

        u = self.grid.u

        # this is the MC limiter from van Leer (1977), as given in 
        # LeVeque (2002).  Note that this is slightly different than
        # the expression from Colella (1990)

        dc = self.grid.scratch_array()
        dl = self.grid.scratch_array()
        dr = self.grid.scratch_array()

        dc[ib:ie+1] = 0.5*(u[ib+1:ie+2] - u[ib-1:ie ])
        dl[ib:ie+1] = u[ib+1:ie+2] - u[ib  :ie+1]
        dr[ib:ie+1] = u[ib  :ie+1] - u[ib-1:ie ]

        # these where's do a minmod()
        d1 = 2.0*numpy.where(numpy.fabs(dl) < numpy.fabs(dr), dl, dr)
        d2 = numpy.where(numpy.fabs(dc) < numpy.fabs(d1), dc, d1)
        ldeltau = numpy.where(dl*dr > 0.0, d2, 0.0)
        
        # now the interface states.  Note that there are 1 more interfaces
        # than zones
        ul = g.scratch_array()
        ur = g.scratch_array()

        ur[ib:ie+1] = u[ib:ie+1] - \
                      0.5*(1.0 + u[ib:ie+1]*dt/self.grid.dx)*ldeltau[ib:ie+1] 

        ul[ib+1:ie+2] = u[ib:ie+1] + \
                        0.5*(1.0 - u[ib:ie+1]*dt/self.grid.dx)*ldeltau[ib:ie+1] 

        return ul, ur


    def riemann(self, ul, ur):
        """ 
        Riemann problem for Burgers' equation.
        """

        S = 0.5*(ul + ur)
        ushock = numpy.where(S > 0.0, ul, ur)
        ushock = numpy.where(S == 0.0, 0.0, ushock)

        # rarefaction solution
        urare = numpy.where(ur <= 0.0, ur, 0.0)
        urare = numpy.where(ul >= 0.0, ul, urare)

        us = numpy.where(ul > ur, ushock, urare)

        return 0.5*us*us


    def update(self, dt, flux):
        """ conservative update """

        g = self.grid

        unew = g.scratch_array()

        unew[g.ilo:g.ihi+1] = g.u[g.ilo:g.ihi+1] + \
            dt/g.dx * (flux[g.ilo:g.ihi+1] - flux[g.ilo+1:g.ihi+2])

        return unew


    def evolve(self, C, tmax):

        self.t = 0.0

        # main evolution loop
        while (self.t < tmax):

            # fill the boundary conditions
            g.fillBCs()

            # get the timestep
            dt = self.timestep(C)

            if (self.t + dt > tmax):
                dt = tmax - self.t

            # get the interface states
            ul, ur = self.states(dt)

            # solve the Riemann problem at all interfaces
            flux = self.riemann(ul, ur)
        
            # do the conservative update
            unew = self.update(dt, flux)

            self.grid.u[:] = unew[:]

            self.t += dt



#-----------------------------------------------------------------------------
# sine

xmin = 0.0
xmax = 1.0
nx = 256
ng = 2
g = Grid1d(nx, ng)

# maximum evolution time based on period for unit velocity
tmax = (xmax - xmin)/1.0

C = 0.8

pylab.clf()

s = Simulation(g)

for i in range(0,10):
    tend = (i+1)*0.02*tmax
    s.initCond("sine")
    s.evolve(C, tend)

    c = 1.0 - (0.1 + i*0.1)
    g = s.grid
    pylab.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=`c`)


g = s.grid
pylab.plot(g.x[g.ilo:g.ihi+1], s.uinit[g.ilo:g.ihi+1], ls=":", color="0.5")

pylab.xlabel("$x$")
pylab.ylabel("$u$")
pylab.savefig("fv-burger-sine.png")
pylab.savefig("fv-burger-sine.eps")


#-----------------------------------------------------------------------------
# rarefaction

xmin = 0.0
xmax = 1.0
nx = 256
ng = 2
g = Grid1d(nx, ng)

# maximum evolution time based on period for unit velocity
tmax = (xmax - xmin)/1.0

C = 0.8

pylab.clf()

s = Simulation(g)

for i in range(0,10):
    tend = (i+1)*0.02*tmax

    s.initCond("rarefaction")
    s.evolve(C, tend)

    c = 1.0 - (0.1 + i*0.1)
    pylab.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=`c`)


pylab.plot(g.x[g.ilo:g.ihi+1], s.uinit[g.ilo:g.ihi+1], ls=":", color="0.5")

pylab.xlabel("$x$")
pylab.ylabel("$u$")

pylab.savefig("fv-burger-rarefaction.png")
pylab.savefig("fv-burger-rarefaction.eps")

