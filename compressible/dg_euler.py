"""
Discontinuous Galerkin for the Euler equations.

Just doing limiting on the conserved variables directly.
"""

import numpy
from numpy.polynomial import legendre
from matplotlib import pyplot
import matplotlib as mpl
import quadpy
from numba import jit
# import tqdm
import riemann

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


class Grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0, m=3):

        assert m > 0

        self.ncons = 3  # 1d hydro, 3 conserved variables
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
        self.u = numpy.zeros((self.ncons, self.m+1, nx+2*ng),
                             dtype=numpy.float64)

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
        self.all_nodes_per_node = numpy.zeros_like(self.u[0])
        for i in range(nx+2*ng):
            self.all_nodes[(m+1)*i:(m+1)*(i+1)] = (self.x[i] +
                                                   self.nodes * self.dx / 2)
            self.all_nodes_per_node[:, i] = (self.x[i] +
                                             self.nodes * self.dx / 2)

    def modal_to_nodal(self, modal):
        for n in range(self.ncons):
            for i in range(self.nx+2*self.ng):
                self.u[n, :, i] = self.V @ modal[n, :, i]
        return self.u

    def nodal_to_modal(self):
        modal = numpy.zeros_like(self.u)
        for n in range(self.ncons):
            for i in range(self.nx+2*self.ng):
                modal[n, :, i] = self.V_inv @ self.u[n, :, i]
        return modal

    def plotting_data(self):
        u_plotting = numpy.zeros((self.ncons, (self.m+1)*(self.nx+2*self.ng)))
        for n in range(self.ncons):
            u_plotting[n, :] = self.u[n, :, :].ravel(order='F')
        return (self.all_nodes,
                u_plotting)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return numpy.zeros((self.ncons, self.m+1, self.nx+2*self.ng),
                           dtype=numpy.float64)

    def fill_BCs(self):
        """ fill all single ghostcell with outflow boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.u[:, :, self.ilo-1-n] = self.u[:, :, self.ilo]
            # right boundary
            self.u[:, :, self.ihi+1+n] = self.u[:, :, self.ihi]


@jit
def euler_flux(u, eos_gamma):
    flux = numpy.zeros_like(u)
    rho = u[0]
    S = u[1]
    E = u[2]
    v = S / rho
    p = (eos_gamma - 1) * (E - rho * v**2 / 2)
    flux[0] = S
    flux[1] = S * v + p
    flux[2] = (E + p) * v
    return flux


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

    def __init__(self, grid, C=0.8, eos_gamma=1.4, limiter=None):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.C = C    # CFL number
        self.eos_gamma = eos_gamma
        self.limiter = limiter  # What it says.

    def init_cond(self, type="sod"):
        """ initialize the data """
        if type == "sod":
            rho_l = 1
            rho_r = 1 / 8
            v_l = 0
            v_r = 0
            p_l = 1
            p_r = 1 / 10
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l**2 / 2)
            E_r = rho_r * (e_r + v_r**2 / 2)
            x = self.grid.all_nodes_per_node
            self.grid.u[0] = numpy.where(x < 0,
                                         rho_l * numpy.ones_like(x),
                                         rho_r * numpy.ones_like(x))
            self.grid.u[1] = numpy.where(x < 0,
                                         S_l * numpy.ones_like(x),
                                         S_r * numpy.ones_like(x))
            self.grid.u[2] = numpy.where(x < 0,
                                         E_l * numpy.ones_like(x),
                                         E_r * numpy.ones_like(x))
        elif type == "advection":
            x = self.grid.all_nodes_per_node
            rho_0 = 1e-3
            rho_1 = 1
            sigma = 0.1
            rho = rho_0 * numpy.ones_like(x)
            rho += (rho_1 - rho_0) * numpy.exp(-(x-0.5)**2/sigma**2)
            v = numpy.ones_like(x)
            p = 1e-6 * numpy.ones_like(x)
            S = rho * v
            e = p / rho / (self.eos_gamma - 1)
            E = rho * (e + v**2 / 2)
            self.grid.u[0, :] = rho[:]
            self.grid.u[1, :] = S[:]
            self.grid.u[2, :] = E[:]
        elif type == "double rarefaction":
            rho_l = 1
            rho_r = 1
            v_l = -2
            v_r = 2
            p_l = 0.4
            p_r = 0.4
            S_l = rho_l * v_l
            S_r = rho_r * v_r
            e_l = p_l / rho_l / (self.eos_gamma - 1)
            e_r = p_r / rho_r / (self.eos_gamma - 1)
            E_l = rho_l * (e_l + v_l**2 / 2)
            E_r = rho_r * (e_r + v_r**2 / 2)
            x = self.grid.all_nodes_per_node
            self.grid.u[0] = numpy.where(x < 0,
                                         rho_l * numpy.ones_like(x),
                                         rho_r * numpy.ones_like(x))
            self.grid.u[1] = numpy.where(x < 0,
                                         S_l * numpy.ones_like(x),
                                         S_r * numpy.ones_like(x))
            self.grid.u[2] = numpy.where(x < 0,
                                         E_l * numpy.ones_like(x),
                                         E_r * numpy.ones_like(x))

    def max_lambda(self):
        rho = self.grid.u[0]
        v = self.grid.u[1] / rho
        p = (self.eos_gamma - 1) * (self.grid.u[2, :] - rho * v**2 / 2)
        cs = numpy.sqrt(self.eos_gamma * p / rho)
        return numpy.max(numpy.abs(v) + cs)

    def timestep(self):
        return self.C * self.grid.dx / self.max_lambda()

    def states(self):
        """ compute the left and right interface states """

        # Evaluate the nodal values at the domain edges
        g = self.grid

        # Extract nodal values
        al = numpy.zeros((g.ncons, g.nx+2*g.ng))
        ar = numpy.zeros((g.ncons, g.nx+2*g.ng))

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            al[:, i] = g.u[:, -1, i-1]
            ar[:, i] = g.u[:, 0, i]

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
            a_zeromode[:, 1:, :] = 0
            for n in range(g.ncons):
                a_cell = (g.V @ a_zeromode[n, :, :])[0, :]
                a_minus = g.u[n, 0, :]
                a_plus = g.u[n, -1, :]
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
                    a1 = factor * a_modal[n, i+1, 1:-1]
                    a2 = theta * (a_modal[n, i, 2:] - a_modal[n, i, 1:-1])
                    a3 = theta * (a_modal[n, i, 1:-1] - a_modal[n, i, :-2])
                    updated_mode[1:-1] = minmod3(a1, a2, a3) / factor
                    did_it_limit = numpy.isclose(a_modal[n, i+1, 1:-1],
                                                 updated_mode[1:-1])
                    a_modal[n, i+1, limiting_todo] = updated_mode[limiting_todo]
                    limiting_todo[1:-1] = numpy.logical_and(limiting_todo[1:-1],
                                              numpy.logical_not(did_it_limit))
            # Get back nodal values
            g.u = g.modal_to_nodal(a_modal)

        return g.u

    def riemann(self, ul, ur, alpha):
        """
        Riemann problem for Burgers using Lax-Friedrichs
        """
        fl = euler_flux(ul, self.eos_gamma)
        fr = euler_flux(ur, self.eos_gamma)
        return ((fl + fr) - (ur - ul)*alpha)/2

    def rk_substep(self, dt):
        """
        Take a single RK substep
        """
        g = self.grid
        g.fill_BCs()
        rhs = g.scratch_array()

        # Integrate flux over element
        f = euler_flux(g.u, self.eos_gamma)
        interior_f = g.scratch_array()
        for n in range(g.ncons):
            interior_f[n, :, :] = g.S.T @ f[n, :, :]
        # Use Riemann solver to get fluxes between elements
        boundary_f = self.riemann(*self.states(), self.max_lambda())
        rhs = interior_f
        rhs[:, 0, 1:-1] += boundary_f[:, 1:-1]
        rhs[:, -1, 1:-1] -= boundary_f[:, 2:]

        # Multiply by mass matrix (inverse).
        rhs_i = g.scratch_array()
        for n in range(g.ncons):
            rhs_i[n, :, :] = 2 / g.dx * g.M_inv @ rhs[n, :, :]

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

    # setup the problem -- Sod
    left = riemann.State(p=1.0, u=0.0, rho=1.0)
    right = riemann.State(p=0.1, u=0.0, rho=0.125)

    rp = riemann.RiemannProblem(left, right)
    rp.find_star_state()

    x_e, rho_e, v_e, p_e = rp.sample_solution(0.2, 1024)
    e_e = p_e / 0.4 / rho_e
    x_e -= 0.5

    # Runs with limiter
    nx = 32
    ng = 1
    xmin = -0.5
    xmax = 0.5
    ms = [1, 3, 5]
    colors = 'brcy'
    for i_m, m in enumerate(ms):
        g = Grid1d(nx, ng, xmin, xmax, m)
        s = Simulation(g, C=0.5/(2*m+1), limiter="moment")
        s.init_cond("sod")
        s.evolve(0.2)
        x, u = g.plotting_data()
        rho = u[0, :]
        v = u[1, :] / u[0, :]
        e = (u[2, :] - rho * v**2 / 2) / rho
        p = (s.eos_gamma - 1) * (u[2, :] - rho * v**2 / 2)
        fig, axes = pyplot.subplots(2, 2)
        axes[0, 0].plot(x[g.ilo:g.ihi+1], rho[g.ilo:g.ihi+1], 'bo')
        axes[0, 0].plot(x_e, rho_e, 'k--')
        axes[0, 1].plot(x[g.ilo:g.ihi+1], v[g.ilo:g.ihi+1], 'bo')
        axes[0, 1].plot(x_e, v_e, 'k--')
        axes[1, 0].plot(x[g.ilo:g.ihi+1], p[g.ilo:g.ihi+1], 'bo')
        axes[1, 0].plot(x_e, p_e, 'k--')
        axes[1, 1].plot(x[g.ilo:g.ihi+1], e[g.ilo:g.ihi+1], 'bo')
        axes[1, 1].plot(x_e, e_e, 'k--')
        axes[1, 0].set_xlabel(r"$x$")
        axes[1, 1].set_xlabel(r"$x$")
        axes[0, 0].set_ylabel(r"$\rho$")
        axes[0, 1].set_ylabel(r"$v$")
        axes[1, 0].set_ylabel(r"$p$")
        axes[1, 1].set_ylabel(r"$e$")
        for ax in axes.flatten():
            ax.set_xlim(-0.5, 0.5)
        fig.tight_layout()
        pyplot.show()

