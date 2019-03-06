"""
Discontinuous Galerkin for Burgers equation.
"""

import numpy
from numpy.polynomial import legendre
from matplotlib import pyplot
import matplotlib as mpl
import quadpy
from numba import jit
# import tqdm

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, m=3):

        assert m > 0

        self.ng = ng
        self.nx = nx
        self.m = m

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
        self.xr = xmin + (numpy.arange(nx+2*ng)-ng+1.0)*self.dx

        # storage for the solution
        # These are the modes of the solution at each point, so the
        # first index is the mode
        # NO! These are actually the *nodal* values
        self.u = numpy.zeros((self.m+1, nx+2*ng), dtype=numpy.float64)

        # Need the Gauss-Lobatto nodes and weights in the reference element
        GL = quadpy.line_segment.GaussLobatto(m+1)
        self.nodes = GL.points
        self.weights = GL.weights
        # To go from modal to nodal we need the Vandermonde matrix
        self.V = legendre.legvander(self.nodes, m)
        c = numpy.eye(m+1)
        # Orthonormalize
        for p in range(m+1):
            self.V[:, p] /= numpy.sqrt(2/(2*p+1))
            c[p, p] /= numpy.sqrt(2/(2*p+1))
        self.V_inv = numpy.linalg.inv(self.V)
        self.M = numpy.linalg.inv(self.V @ self.V.T)
        self.M_inv = self.V @ self.V.T
        # Derivatives of Legendre polynomials lead to derivatives of V
        dV = legendre.legval(self.nodes,
                             legendre.legder(c)).T
        self.D = dV @ self.V_inv
        # Stiffness matrix for the interior flux
        self.S = self.M @ self.D

        # Nodes in the computational coordinates
        self.all_nodes = numpy.zeros((m+1)*(nx+2*ng), dtype=numpy.float64)
        self.all_nodes_per_node = numpy.zeros_like(self.u)
        for i in range(nx+2*ng):
            self.all_nodes[(m+1)*i:(m+1)*(i+1)] = (self.x[i] +
                                                   self.nodes * self.dx / 2)
            self.all_nodes_per_node[:, i] = (self.x[i] +
                                             self.nodes * self.dx / 2)

    def modal_to_nodal(self, modal):
        for i in range(self.nx+2*self.ng):
            self.u[:, i] = self.V @ modal[:, i]
        return self.u

    def nodal_to_modal(self):
        modal = numpy.zeros_like(self.u)
        for i in range(self.nx+2*self.ng):
            modal[:, i] = self.V_inv @ self.u[:, i]
        return modal

    def plotting_data(self):
        return (self.all_nodes,
                self.u.ravel(order='F'))

    def plotting_data_high_order(self, npoints=50):
        assert npoints > 2
        p_nodes = numpy.zeros(npoints*(self.nx+2*self.ng), dtype=numpy.float64)
        p_data = numpy.zeros_like(p_nodes)
        for i in range(self.nx+2*self.ng):
            p_nodes[npoints*i:npoints*(i+1)] = numpy.linspace(self.xl[i],
                                                              self.xr[i],
                                                              npoints)
            modal = self.V_inv @ self.u[:, i]
            for p in range(self.m+1):
                modal[p] /= numpy.sqrt(2/(2*p+1))
            scaled_x = 2 * (p_nodes[npoints*i:npoints*(i+1)] -
                            self.x[i]) / self.dx
            p_data[npoints*i:npoints*(i+1)] = legendre.legval(scaled_x, modal)
        return p_nodes, p_data

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.m+1, self.nx+2*self.ng), dtype=numpy.float64)

    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.u[:, self.ilo-1-n] = self.u[:, self.ihi-n]
            # right boundary
            self.u[:, self.ihi+1+n] = self.u[:, self.ilo+n]

    def norm(self, e):
        """
        Return the norm of quantity e which lives on the grid.

        This is the 'broken norm': the quantity is integrated over each
        individual element using Gauss-Lobatto quadrature (as we have those
        nodes and weights), and the 2-norm of the result is then returned.
        """
        if not numpy.allclose(e.shape, self.all_nodes_per_node.shape):
            return None

        # This is actually a pointwise norm, not quadrature'd
        return numpy.sqrt(self.dx*numpy.sum(e[:, self.ilo:self.ihi+1]**2))


@jit
def burgers_flux(u):
    return u**2/2


@jit
def minmod3(a1, a2, a3):
    """
    Utility function that does minmod on three inputs
    """
    signs1 = a1 * a2
    signs2 = a1 * a3
    signs3 = a2 * a3
    same_sign = numpy.logical_and(numpy.logical_and(signs1 > 0,
                                                    signs2 > 0),
                                  signs3 > 0)
    minmod = numpy.min(numpy.abs(numpy.vstack((a1, a2, a3))),
                       axis=0) * numpy.sign(a1)

    return numpy.where(same_sign, minmod, numpy.zeros_like(a1))


@jit
def limit(g, limiter):
    """
    After evolution, limit the slopes.
    """

    # Limiting!

    if limiter == "moment":

        # Limiting, using moment limiting (Hesthaven p 445-7)
        theta = 2
        a_modal = g.nodal_to_modal()
        # First, work out where limiting is needed
        limiting_todo = numpy.ones(g.nx+2*g.ng, dtype=bool)
        limiting_todo[:g.ng] = False
        limiting_todo[-g.ng:] = False
        # Get the cell average and the nodal values at the boundaries
        a_zeromode = a_modal.copy()
        a_zeromode[1:, :] = 0
        a_cell = (g.V @ a_zeromode)[0, :]
        a_minus = g.u[0, :]
        a_plus = g.u[-1, :]
        # From the cell averages and boundary values we can construct
        # alternate values at the boundaries
        a_left = numpy.zeros(g.nx+2*g.ng)
        a_right = numpy.zeros(g.nx+2*g.ng)
        a_left[1:-1] = a_cell[1:-1] - minmod3(a_cell[1:-1] - a_minus[1:-1],
                                              a_cell[1:-1] - a_cell[:-2],
                                              a_cell[2:] - a_cell[1:-1])
        a_right[1:-1] = a_cell[1:-1] + minmod3(a_plus[1:-1] - a_cell[1:-1],
                                               a_cell[1:-1] - a_cell[:-2],
                                               a_cell[2:] - a_cell[1:-1])
        limiting_todo[1:-1] = numpy.logical_not(
                numpy.logical_and(numpy.isclose(a_left[1:-1],
                                                a_minus[1:-1]),
                                  numpy.isclose(a_right[1:-1],
                                                a_plus[1:-1])))
        # Now, do the limiting. Modify moments where needed, and as soon as
        # limiting isn't needed, stop
        updated_mode = numpy.zeros(g.nx+2*g.ng)
        for i in range(g.m-1, -1, -1):
            factor = numpy.sqrt((2*i+3)*(2*i+5))
            a1 = factor * a_modal[i+1, 1:-1]
            a2 = theta * (a_modal[i, 2:] - a_modal[i, 1:-1])
            a3 = theta * (a_modal[i, 1:-1] - a_modal[i, :-2])
            updated_mode[1:-1] = minmod3(a1, a2, a3) / factor
            did_it_limit = numpy.isclose(a_modal[i+1, 1:-1],
                                         updated_mode[1:-1])
            a_modal[i+1, limiting_todo] = updated_mode[limiting_todo]
            limiting_todo[1:-1] = numpy.logical_and(limiting_todo[1:-1],
                                      numpy.logical_not(did_it_limit))
        # Get back nodal values
        limited_a = g.modal_to_nodal(a_modal)

    else:
        limited_a = g.u

    return limited_a


class Simulation(object):

    def __init__(self, grid, C=0.8, limiter=None):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C    # CFL number
        self.limiter = limiter  # What it says.

    def init_cond(self, type="sine"):
        """ initialize the data """
        if type == "tophat":
            def init_u(x):
                return numpy.where(numpy.logical_and(x >= 0.333,
                                                     x <= 0.666),
                                   numpy.ones_like(x),
                                   numpy.zeros_like(x))
        elif type == "sine":
            def init_u(x):
                return numpy.where(numpy.logical_and(x >= 0.333,
                                                     x <= 0.666),
                                   numpy.ones_like(x) +
                                   0.5 * numpy.sin(
                                           2.0 * numpy.pi *
                                           (x - 0.333) / 0.333),
                                   numpy.ones_like(x))
        elif type == "smooth_sine":
            def init_u(x):
                return numpy.sin(2.0 * numpy.pi * x /
                                 (self.grid.xmax - self.grid.xmin))
        elif type == "gaussian":
            def init_u(x):
                local_xl = x - self.grid.dx/2
                local_xr = x + self.grid.dx/2
                al = 1.0 + numpy.exp(-60.0*(local_xl - 0.5)**2)
                ar = 1.0 + numpy.exp(-60.0*(local_xr - 0.5)**2)
                ac = 1.0 + numpy.exp(-60.0*(x - 0.5)**2)

                return (1./6.)*(al + 4*ac + ar)

        self.grid.u = init_u(self.grid.all_nodes_per_node)

    def timestep(self):
        """ return the advective timestep """
        return self.C*self.grid.dx/numpy.max(numpy.abs(
                self.grid.u[:, self.grid.ilo:self.grid.ihi+1]))

    def states(self):
        """ compute the left and right interface states """

        # Evaluate the nodal values at the domain edges
        g = self.grid

        # Extract nodal values

        al = numpy.zeros(g.nx+2*g.ng)
        ar = numpy.zeros(g.nx+2*g.ng)

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            al[i] = g.u[-1, i-1]
            ar[i] = g.u[0, i]

        return al, ar

    def limit(self):
        """
        After evolution, limit the slopes.
        """

        # Evaluate the nodal values at the domain edges
        g = self.grid

        # Limiting!

        if self.limiter == "moment":

            # Limiting, using moment limiting (Hesthaven p 445-7)
            theta = 2
            a_modal = g.nodal_to_modal()
            # First, work out where limiting is needed
            limiting_todo = numpy.ones(g.nx+2*g.ng, dtype=bool)
            limiting_todo[:g.ng] = False
            limiting_todo[-g.ng:] = False
            # Get the cell average and the nodal values at the boundaries
            a_zeromode = a_modal.copy()
            a_zeromode[1:, :] = 0
            a_cell = (g.V @ a_zeromode)[0, :]
            a_minus = g.u[0, :]
            a_plus = g.u[-1, :]
            # From the cell averages and boundary values we can construct
            # alternate values at the boundaries
            a_left = numpy.zeros(g.nx+2*g.ng)
            a_right = numpy.zeros(g.nx+2*g.ng)
            a_left[1:-1] = a_cell[1:-1] - minmod3(a_cell[1:-1] - a_minus[1:-1],
                                                  a_cell[1:-1] - a_cell[:-2],
                                                  a_cell[2:] - a_cell[1:-1])
            a_right[1:-1] = a_cell[1:-1] + minmod3(a_plus[1:-1] - a_cell[1:-1],
                                                   a_cell[1:-1] - a_cell[:-2],
                                                   a_cell[2:] - a_cell[1:-1])
            limiting_todo[1:-1] = numpy.logical_not(
                    numpy.logical_and(numpy.isclose(a_left[1:-1],
                                                    a_minus[1:-1]),
                                      numpy.isclose(a_right[1:-1],
                                                    a_plus[1:-1])))
            # Now, do the limiting. Modify moments where needed, and as soon as
            # limiting isn't needed, stop
            updated_mode = numpy.zeros(g.nx+2*g.ng)
            for i in range(g.m-1, -1, -1):
                factor = numpy.sqrt((2*i+3)*(2*i+5))
                a1 = factor * a_modal[i+1, 1:-1]
                a2 = theta * (a_modal[i, 2:] - a_modal[i, 1:-1])
                a3 = theta * (a_modal[i, 1:-1] - a_modal[i, :-2])
                updated_mode[1:-1] = minmod3(a1, a2, a3) / factor
                did_it_limit = numpy.isclose(a_modal[i+1, 1:-1],
                                             updated_mode[1:-1])
                a_modal[i+1, limiting_todo] = updated_mode[limiting_todo]
                limiting_todo[1:-1] = numpy.logical_and(limiting_todo[1:-1],
                                          numpy.logical_not(did_it_limit))
            # Get back nodal values
            g.u = g.modal_to_nodal(a_modal)

        return g.u

    def riemann(self, ul, ur, alpha):
        """
        Riemann problem for Burgers using Lax-Friedrichs
        """

        return ((burgers_flux(ul) + burgers_flux(ur)) - (ur - ul)*alpha)/2

    def rk_substep(self, dt):
        """
        Take a single RK substep
        """
        g = self.grid
        g.fill_BCs()
        rhs = g.scratch_array()

        # Integrate flux over element
        f = burgers_flux(g.u)
        interior_f = g.S.T @ f
        # Use Riemann solver to get fluxes between elements
        boundary_f = self.riemann(*self.states(), numpy.max(numpy.abs(g.u)))
        rhs = interior_f
        rhs[0, 1:-1] += boundary_f[1:-1]
        rhs[-1, 1:-1] -= boundary_f[2:]

        # Multiply by mass matrix (inverse).
        rhs_i = 2 / g.dx * g.M_inv @ rhs

        return rhs_i

    def evolve(self, tmax):
        """ evolve Burgers equation using RK3 (SSP) """
        self.t = 0.0
        g = self.grid

        # main evolution loop
#        pbar = tqdm.tqdm(total=100)
        while self.t < tmax:
            # pbar.update(100*self.t/tmax)
            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK3 - SSP
            # Store the data at the start of the step
            a_start = g.u.copy()
            k1 = dt * self.rk_substep(dt)
            g.u = a_start + k1
            g.u = self.limit()
            g.fill_BCs()
            a1 = g.u.copy()
            k2 = dt * self.rk_substep(dt)
            g.u = (3 * a_start + a1 + k2) / 4
            g.u = self.limit()
            g.fill_BCs()
            a2 = g.u.copy()
            k3 = dt * self.rk_substep(dt)
            g.u = (a_start + 2 * a2 + 2 * k3) / 3
            g.u = self.limit()
            g.fill_BCs()

            self.t += dt
#        pbar.close()


if __name__ == "__main__":

    # Runs with limiter
    nx = 128
    ng = 1
    xmin = 0
    xmax = 1
    ms = [1, 3, 5]
    colors = 'brcy'
    for i_m, m in enumerate(ms):
        g = Grid1d(nx, ng, xmin, xmax, m)
        s = Simulation(g, C=0.5/(2*m+1), limiter="moment")
        s.init_cond("sine")
        x, u = g.plotting_data()
        x_start = x.copy()
        u_start = u.copy()
        s.evolve(0.2)
        x_end, u_end = g.plotting_data()
        pyplot.plot(x_end.copy(), u_end.copy(), f'{colors[i_m]}-',
                    label=rf'$m={m}$')
    pyplot.plot(x_start, u_start, 'k--', alpha=0.5, label='Initial data')
    pyplot.xlim(0, 1)
    pyplot.legend()
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$u$')
    pyplot.show()
