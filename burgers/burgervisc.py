"""
solve the viscous Burger's equation: u_t + u u_x = nu u_xx

we use an explicit piecewise linear finite-volume method to evaluate
the advective flux and then discretize the diffusion part implicitly
(Crank-Nicolson) with the advective piece as a source to update in
time.

The resulting method is second-order in space and time.

M. Zingale (2013-04-03)
"""

import numpy
from scipy import linalg
import sys
import pylab
import math

def diffuse(gr, u, nu, S, dt):
    """ diffuse u implicitly (C-N) through timestep dt with source S """

    unew = gr.scratchArray()
    
    alpha = nu*dt/gr.dx**2

    # create the RHS of the matrix
    R = 0.5*nu*dt*lap(gr,u)
    R = R[gr.ilo:gr.ihi+1]
    R += u[gr.ilo:gr.ihi+1]
    R += dt*S[gr.ilo:gr.ihi+1]
    
    # create the diagonal, d+1 and d-1 parts of the matrix
    d = (1.0 + alpha)*numpy.ones(gr.nx)
    u = -0.5*alpha*numpy.ones(gr.nx)
    u[0] = 0.0

    l = -0.5*alpha*numpy.ones(gr.nx)
    l[gr.nx-1] = 0.0

    # set the boundary conditions by changing the matrix elements

    # homogeneous neumann
    d[0] = 1.0 + 0.5*alpha
    d[gr.nx-1] = 1.0 + 0.5*alpha

    # dirichlet
    #d[0] = 1.0 + 1.5*alpha
    #R[0] += alpha*0.0

    #d[gr.nx-1] = 1.0 + 1.5*alpha
    #R[gr.nx-1] += alpha*0.0

    # solve
    A = numpy.matrix([u,d,l])
    unew[gr.ilo:gr.ihi+1] = linalg.solve_banded((1,1), A, R)

    return unew


def advect(gr, u, S, dt):
    """ compute the advective term that updates u in time.  Here, S is
        a source term """

    # we need the states in the first ghost cell too
    ib = gr.ilo-1
    ie = gr.ihi+1

    # compute the limited slopes -- 2nd order MC limiter
    test = gr.scratchArray()
    test[ib:ie+1] = (u[ib+1:ie+2] - u[ib:ie+1])*(u[ib:ie+1] - u[ib-1:ie])

    dc = gr.scratchArray()
    dl = gr.scratchArray()
    dr = gr.scratchArray()

    dc[ib:ie+1] = 0.5*numpy.fabs(u[ib+1:ie+2] - u[ib-1:ie ])
    dl[ib:ie+1] = 0.5*numpy.fabs(u[ib+1:ie+2] - u[ib  :ie+1])
    dr[ib:ie+1] = 0.5*numpy.fabs(u[ib  :ie+1] - u[ib-1:ie ])
        
    minslope = numpy.minimum(dc, numpy.minimum(2.0*dl, 2.0*dr))
    ldeltau = numpy.where(test > 0.0, minslope, 0.0)*numpy.sign(dc)

    # construct the interface states
    ul = gr.scratchArray()
    ur = gr.scratchArray()

    ur[ib:ie+1] = u[ib:ie+1] - \
        0.5*(1.0 + u[ib:ie+1]*dt/gr.dx)*ldeltau[ib:ie+1] + 0.5*dt*S[ib:ie+1]

    ul[ib+1:ie+2] = u[ib:ie+1] + \
        0.5*(1.0 - u[ib:ie+1]*dt/gr.dx)*ldeltau[ib:ie+1] + 0.5*dt*S[ib:ie+1]

    # Riemann problem -- Burger's Eq

    # shock speed and shock state
    S = 0.5*(ul + ur)
    ushock = numpy.where(S > 0.0, ul, ur)
    ushock = numpy.where(S == 0.0, 0.0, ushock)

    # rarefaction solution
    urare = numpy.where(ur <= 0.0, ur, 0.0)
    urare = numpy.where(ul >= 0.0, ul, urare)

    us = numpy.where(ul > ur, ushock, urare)

    # construct the advective update
    F = 0.5*us*us
    A = gr.scratchArray()

    A[ib:ie+1] = (F[ib:ie+1] - F[ib+1:ie+2])/gr.dx

    return A


def lap(gr, u):
    """ compute the Laplacian of u, including the first ghost cells """

    lapu = gr.scratchArray()

    ib = gr.ilo-1
    ie = gr.ihi+1

    lapu[ib:ie+1] = (u[ib-1:ie] - 2.0*u[ib:ie+1] + u[ib+1:ie+2])/gr.dx**2

    return lapu


def estDt(gr, cfl, u):
    """ estimate the timestep """

    # use the proported flame speed
    dt = cfl*gr.dx/numpy.max(numpy.abs(u))
    return dt


class grid:

    def __init__(self, nx, ng=1, xmin=0.0, xmax=1.0, vars=None):
        """ grid class initialization """
        
        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin)/nx
        self.x = (numpy.arange(nx+2*ng) + 0.5 - ng)*self.dx + xmin

        self.ilo = ng
        self.ihi = ng+nx-1

        self.data = {}

        for v in vars:
            self.data[v] = numpy.zeros((2*ng+nx), dtype=numpy.float64)


    def fillBC(self, var):

        if not var in self.data.keys():
            sys.exit("invalid variable")

        vp = self.data[var]

        # Neumann BCs
        vp[0:self.ilo+1] = vp[self.ilo]
        vp[self.ihi+1:] = vp[self.ihi]


    def scratchArray(self):
        return numpy.zeros((2*self.ng+self.nx), dtype=numpy.float64)


    def initialize(self):
        """ initial conditions """

        u = self.data["u"]
        u[:] = 0.0
        u[self.nx/2-0.15*self.nx:self.nx/2+0.15*self.nx+1] = 1.0

        #u[:] = 1.0 + 0.5*numpy.sin(2.0*3*math.pi*self.x)*numpy.exp(-(self.x-0.5)**2/0.1**2)
        index = numpy.logical_and(self.x >= 0.333, self.x <= 0.666)
        u[:] = 1.0
        u[index] += 0.5*numpy.sin(2.0*math.pi*(self.x[index]-0.333)/0.333)


def evolve(nx, nu, cfl, tmax, dovis=0):
    """ 
    the main evolution loop.  Evolve 
  
     phi_t + u u_x = nu u_xx

    from t = 0 to tmax
    """

    # create the grid -- two ghostcells for advection
    gr = grid(nx, ng=2, xmin = 0.0, xmax=1.0,
              vars=["u"])

    # pointers to the data at various stages
    u  = gr.data["u"]

    # initialize
    gr.initialize()
    gr.fillBC("u")

    # runtime plotting
    if dovis == 1:
        pylab.ion()
    
    t = 0.0
    while (t < tmax):

        dt = estDt(gr, cfl, u)

        if (t + dt > tmax):
            dt = tmax - t

        # construct the explicit diffusion source
        S = nu*lap(gr, u)

        # construct the advective update
        A = advect(gr, u, S, dt)

        # diffuse for dt
        unew = diffuse(gr, u, nu, A, dt)
        #unew = u + dt*A

        u[:] = unew[:]
        gr.fillBC("u")

        t += dt

        if dovis == 1:
            pylab.clf()
            pylab.plot(gr.x, u)
            pylab.xlim(gr.xmin,gr.xmax)
            pylab.ylim(0.0,2.0)
            pylab.title("t = %f" % (t))
            pylab.draw()

    print t
    return u, gr.x






nx = 256
nu1 = 0.005
cfl = 0.8
tmax = 0.2

u1, x1 = evolve(nx, nu1, cfl, tmax, dovis=0)

nu2 = 0.0005
u2, x2 = evolve(nx, nu2, cfl, tmax, dovis=0)

nu3 = 0.00005
u3, x3 = evolve(nx, nu3, cfl, tmax, dovis=0)

pylab.plot(x1, u1, label=r"$\nu = %f$" % (nu1))
pylab.plot(x2, u2, label=r"$\nu = %f$" % (nu2), ls="--", color="k")
pylab.plot(x3, u3, label=r"$\nu = %f$" % (nu3), ls=":", color="k")

pylab.legend(frameon=False)

pylab.xlim(0.0, 1.0)

pylab.tight_layout()
pylab.savefig("burgervisc.png")
pylab.savefig("burgervisc.eps")
