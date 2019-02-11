"""
Discontinuous Galerkin for the advection equation.
"""

import numpy
from matplotlib import pyplot
import matplotlib as mpl
import quadpy

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'



class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, np=3):

        self.ng = ng
        self.nx = nx
        self.np = np

        self.xmin = xmin
        self.xmax = xmax

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x = xmin + (numpy.arange(nx+2*ng)-ng+0.5)*self.dx
        self.xl = xmin + (numpy.arange(nx+2*ng)-ng)*self.dx
        self.xr = xmin + (numpy.arange(nx+2*ng)+1.0)*self.dx

        # storage for the solution
        # These are the modes of the solution at each point, so the
        # first index is the mode
        self.a = numpy.zeros((self.np, nx+2*ng), dtype=numpy.float64)
        
        # Need the Gauss-Lobatto nodes and weights in the reference element
        GL = quadpy.line_segment.GaussLobatto(np+2)
        self.nodes = GL.points
        self.weights = GL.weights
        # To go from modal to nodal we need the Vandermonde matrix
        self.V = numpy.zeros((np+2, np+2))
        for n in range(np+2):
            c = numpy.zeros(n)
            c[n] = 1
            self.V[:, n] = numpy.polynomial.legendre.legval(self.nodes, c)
        # We can now get the nodal values from self.V @ self.a[:, i]
        
        # Need the weights multiplied by P_p' for the interior flux
        self.modified_weights = numpy.zeros((np, np+2))
        for p in range(np):
            pp_c = numpy.zeros(p+1)
            pp_c[p] = 1
            pp_prime_c = numpy.polynomial.legendre.legder(pp_c)
            pp_prime_nodes = numpy.polynomial.legendre.legval(self.nodes, pp_prime_c)
            self.modified_weights[p, :] = self.weights * pp_prime_nodes
        
        # Nodes in the computational coordinates
        self.all_nodes = numpy.zeros((np+2)*(nx+2*ng), dtype=numpy.float64)
        for i in range(nx+2*ng):
            self.all_nodes[(np+2)*i: (np+2)*(i+1)] = self.x + self.nodes * self.dx / 2

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.np, self.nx+2*self.ng), dtype=numpy.float64)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.a[:, self.ilo-1-n] = self.a[:, self.ihi-n]

            # right boundary
            self.a[:, self.ihi+1+n] = self.a[:, self.ilo+n]

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if len(e) != 2*self.ng + self.nx:
            return None

        #return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))
        return numpy.max(abs(e[0, self.ilo:self.ihi+1]))


class Simulation(object):

    def __init__(self, grid, u, C=0.8):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.u = u   # the constant advective velocity
        self.C = C   # CFL number


    def init_cond(self, type="sine"):
        """ initialize the data """
        if type == "tophat":
            init_a = lambda x : numpy.where(numpy.logical_and(x >=0.333, x <=0.666), 
                                            numpy.ones_like(x), numpy.zeros_like(x))

        elif type == "sine":
            init_a = lambda x : numpy.sin(2.0*numpy.pi*x/(self.grid.xmax-self.grid.xmin))

        nodal_a = init_a(self.grid.all_nodes)
        for i in range(self.grid.np+2*self.grid.ng):
            for p in range(self.grid.np):
                self.grid.a[p, i] = numpy.sum(nodal_a[(self.grid.np+2)*i:(self.grid.np+2)*(i+1)] * self.V[:,p])


    def timestep(self):
        """ return the advective timestep """
        return self.C*self.grid.dx/self.u


    def period(self):
        """ return the period for advection with velocity u """
        return (self.grid.xmax - self.grid.xmin)/self.u


    def states(self):
        """ compute the left and right interface states """

        # Evaluate the nodal values at the domain edges
        g = self.grid

        al = g.scratch_array()
        ar = g.scratch_array()

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            for p in range(g.np):
                al[p, i] = numpy.dot(g.a[:, i-1], g.V[-1, :])
                ar[p, i] = numpy.dot(g.a[:, i  ], g.V[ 0, :])

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


    def rk_substep(self):
        
        g = self.grid
        g.fill_BCs()
        rhs = g.scratch_array()
        
        # Integrate flux over element
        interior_f = g.scratch_array()
        for p in range(g.np):
            for i in range(g.ilo, g.ihi+1):
                nodal_a = g.V @ g.a[:, i]
                nodal_f = self.u * nodal_a
                interior_f[p, i] = numpy.dot(nodal_f, g.modified_weights[p, :])
        
        # Use Riemann solver to get fluxes between elements
        boundary_f = self.riemann(*self.states())
        rhs = interior_f
        for p in range(g.np):
            for i in range(g.ilo, g.ihi+1):
                rhs[p, i] += (-1)**p * boundary_f[p, i] - boundary_f[p, i+1]
        
        # Multiply by mass matrix, which is diagonal.
        for p in range(g.np):
            rhs[p, :] *= (2*p + 1) / 2
        
        return rhs


    def evolve(self, num_periods=1):
        """ evolve the linear advection equation using RK3 (SSP) """
        self.t = 0.0
        g = self.grid

        tmax = num_periods*self.period()

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK3 - SSP
            # Store the data at the start of the step
            a_start = g.a.copy()
            k1 = dt * self.rk_substep()
            g.a = a_start + k1
            # a1 = g.a.copy()
            k2 = dt * self.rk_substep()
            g.a = a_start + k2 / 4
            a2 = g.a.copy()
            k3 = dt * self.rk_substep()
            g.a = (a_start + 2 * a2 + 2 * k3) / 3

            self.t += dt
