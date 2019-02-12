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

        assert np > 1

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
        GL = quadpy.line_segment.GaussLobatto(np)
        self.nodes = GL.points
        self.weights = GL.weights
        # To go from modal to nodal we need the Vandermonde matrix
        self.V = numpy.zeros((np, np))
        c = numpy.eye(np)
        self.V = numpy.polynomial.legendre.legval(self.nodes, c)
        self.V_inv = numpy.linalg.inv(self.V)
        self.M_inv = 2 / self.dx * self.V @ self.V.T
        print("Inverse mass matrix is", self.M_inv)

        # Need the weights multiplied by P_p' for the interior flux
        self.modified_weights = numpy.zeros((np, np))
        for p in range(np):
            pp_c = numpy.zeros(p+1)
            pp_c[p] = 1
            pp_prime_c = numpy.polynomial.legendre.legder(pp_c)
            pp_prime_nodes = numpy.polynomial.legendre.legval(self.nodes,
                                                              pp_prime_c)
            self.modified_weights[p, :] = self.weights * pp_prime_nodes

        # Nodes in the computational coordinates
        self.all_nodes = numpy.zeros((np)*(nx+2*ng), dtype=numpy.float64)
        self.all_nodes_per_node = numpy.zeros_like(self.a)
        for i in range(nx+2*ng):
            self.all_nodes[(np)*i:(np)*(i+1)] = (self.x[i] +
                                                 self.nodes * self.dx / 2)
            self.all_nodes_per_node[:, i] = (self.x[i] +
                                             self.nodes * self.dx / 2)

    def modal_to_nodal(self):
        nodal = numpy.zeros_like(self.a)
        for i in range(self.nx+2*self.ng):
            nodal[:, i] = self.V @ self.a[:, i]
        return nodal

    def nodal_to_modal(self, nodal):
        for i in range(self.nx+2*self.ng):
            self.a[:, i] = self.V_inv @ nodal[:, i]

    def plotting_data(self):
        return (self.all_nodes,
                self.modal_to_nodal().ravel(order='F'))

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

        # return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))
        return numpy.max(abs(e[0, self.ilo:self.ihi+1]))


class Simulation(object):

    def __init__(self, grid, u, C=0.8):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.u = u    # the constant advective velocity
        self.C = C    # CFL number

    def init_cond(self, type="sine"):
        """ initialize the data """
        if type == "tophat":
            def init_a(x):
                return numpy.where(numpy.logical_and(x >= 0.333,
                                                     x <= 0.666),
                                   numpy.ones_like(x),
                                   numpy.zeros_like(x))

        elif type == "sine":
            def init_a(x):
                return numpy.sin(2.0 * numpy.pi * x /
                                 (self.grid.xmax - self.grid.xmin))

        nodal_a = init_a(self.grid.all_nodes_per_node)
        self.grid.nodal_to_modal(nodal_a)

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

        al = numpy.zeros(g.nx+2*g.ng)
        ar = numpy.zeros(g.nx+2*g.ng)

        nodal = g.modal_to_nodal()

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            al[i] = nodal[-1, i-1]
            ar[i] = nodal[ 0, i  ]

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
        """
        Take a single RK substep
        """
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
                rhs[p, i] += (-1)**p * boundary_f[i] - boundary_f[i+1]

        # Multiply by mass matrix (inverse), which is diagonal.
        # Is it orthonormal?
#        for p in range(g.np):
#            rhs[p, :] *= (2*p + 1) / 2
#        for p in range(g.np):
#            rhs[p, :] *= (2 * p + 1) / g.dx**p
        rhs = g.M_inv @ rhs

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


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # DG of sine wave

    xmin = 0.0
    xmax = 1.0
    nx = 4
    ng = 1

    g3 = Grid1d(nx, ng, xmin=xmin, xmax=xmax, np=3)
    g7 = Grid1d(nx, ng, xmin=xmin, xmax=xmax, np=7)

    u = 1.0

    # The CFL limit for DG is reduced by a factor 1/(2 p + 1)
    s3 = Simulation(g3, u, C=0.8/(2*3+1))
    s3.init_cond("sine")
    s7 = Simulation(g7, u, C=0.8/(2*7+1))
    s7.init_cond("sine")
    # Plot the initial data to show how, difference in nodal locations as
    # number of modes varies
    plot_x3, plot_a3 = g3.plotting_data()
    a3init = plot_a3.copy()
    plot_x7, plot_a7 = g7.plotting_data()
    a7init = plot_a7.copy()
    pyplot.plot(plot_x3, plot_a3, 'bo', label=r"$p=3$")
    pyplot.plot(plot_x7, plot_a7, 'r^', label=r"$p=7$")
    pyplot.xlim(0, 1)
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$a$')
    pyplot.legend()
    pyplot.show()

    s3.evolve(num_periods=1)
    plot_x3, plot_a3 = g3.plotting_data()
    pyplot.plot(plot_x3, plot_a3, 'bo', label=r"$p=3$")
    pyplot.plot(plot_x3, a3init, 'r^', label=r"$p=3$, initial")
    pyplot.xlim(0, 1)
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$a$')
    pyplot.legend()
    pyplot.show()

    s7.evolve(num_periods=1)
    plot_x7, plot_a7 = g7.plotting_data()
    pyplot.plot(plot_x7, plot_a7, 'bo', label=r"$p=7$")
    pyplot.plot(plot_x7, a7init, 'r^', label=r"$p=7$, initial")
    pyplot.xlim(0, 1)
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$a$')
    pyplot.legend()
    pyplot.show()
