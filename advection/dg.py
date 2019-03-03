"""
Discontinuous Galerkin for the advection equation.
"""

import numpy
from numpy.polynomial import legendre
from matplotlib import pyplot
import matplotlib as mpl
import quadpy
from scipy.integrate import ode

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
        self.a = numpy.zeros((self.m+1, nx+2*ng), dtype=numpy.float64)

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
        self.all_nodes_per_node = numpy.zeros_like(self.a)
        for i in range(nx+2*ng):
            self.all_nodes[(m+1)*i:(m+1)*(i+1)] = (self.x[i] +
                                                   self.nodes * self.dx / 2)
            self.all_nodes_per_node[:, i] = (self.x[i] +
                                             self.nodes * self.dx / 2)

    def modal_to_nodal(self, modal):
        for i in range(self.nx+2*self.ng):
            self.a[:, i] = self.V @ modal[:, i]
        return self.a

    def nodal_to_modal(self):
        modal = numpy.zeros_like(self.a)
        for i in range(self.nx+2*self.ng):
            modal[:, i] = self.V_inv @ self.a[:, i]
        return modal

    def plotting_data(self):
        return (self.all_nodes,
                self.a.ravel(order='F'))

    def plotting_data_high_order(self, npoints=50):
        assert npoints > 2
        p_nodes = numpy.zeros(npoints*(self.nx+2*self.ng), dtype=numpy.float64)
        p_data = numpy.zeros_like(p_nodes)
        for i in range(self.nx+2*self.ng):
            p_nodes[npoints*i:npoints*(i+1)] = numpy.linspace(self.xl[i],
                                                              self.xr[i],
                                                              npoints)
            modal = self.V_inv @ self.a[:, i]
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
            self.a[:, self.ilo-1-n] = self.a[:, self.ihi-n]
            # right boundary
            self.a[:, self.ihi+1+n] = self.a[:, self.ilo+n]

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


class Simulation(object):

    def __init__(self, grid, u, C=0.8, limiter=None):
        self.grid = grid
        self.t = 0.0  # simulation time
        self.u = u    # the constant advective velocity
        self.C = C    # CFL number
        self.limiter = limiter  # What it says.

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
        elif type == "gaussian":
            def init_a(x):
                local_xl = x - self.grid.dx/2
                local_xr = x + self.grid.dx/2
                al = 1.0 + numpy.exp(-60.0*(local_xl - 0.5)**2)
                ar = 1.0 + numpy.exp(-60.0*(local_xr - 0.5)**2)
                ac = 1.0 + numpy.exp(-60.0*(x - 0.5)**2)

                return (1./6.)*(al + 4*ac + ar)

        self.grid.a = init_a(self.grid.all_nodes_per_node)

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

        # Extract nodal values

        al = numpy.zeros(g.nx+2*g.ng)
        ar = numpy.zeros(g.nx+2*g.ng)

        # i is looping over interfaces, so al is the right edge of the left
        # element, etc.
        for i in range(g.ilo, g.ihi+2):
            al[i] = g.a[-1, i-1]
            ar[i] = g.a[0, i]

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
            a_minus = g.a[0, :]
            a_plus = g.a[-1, :]
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
            g.a = g.modal_to_nodal(a_modal)

        return g.a

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
        f = self.u * g.a
        interior_f = g.S.T @ f
        # Use Riemann solver to get fluxes between elements
        boundary_f = self.riemann(*self.states())
        rhs = interior_f
        rhs[0, 1:-1] += boundary_f[1:-1]
        rhs[-1, 1:-1] -= boundary_f[2:]

        # Multiply by mass matrix (inverse).
        rhs_i = 2 / g.dx * g.M_inv @ rhs

        return rhs_i

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
            g.a = self.limit()
            a1 = g.a.copy()
            k2 = dt * self.rk_substep()
            g.a = (3 * a_start + a1 + k2) / 4
            g.a = self.limit()
            a2 = g.a.copy()
            k3 = dt * self.rk_substep()
            g.a = (a_start + 2 * a2 + 2 * k3) / 3
            g.a = self.limit()

            self.t += dt

    def evolve_scipy(self, num_periods=1):
        """ evolve the linear advection equation using scipy """
        self.t = 0.0
        g = self.grid

        def rk_substep_scipy(t, y):
            local_a = numpy.reshape(y, g.a.shape)
            # Periodic BCs
            local_a[:, :g.ng] = local_a[:, -2*g.ng:-g.ng]
            local_a[:, -g.ng:] = local_a[:, g.ng:2*g.ng]
            # Integrate flux over element
            f = self.u * local_a
            interior_f = g.S.T @ f
            # Use Riemann solver to get fluxes between elements
            al = numpy.zeros(g.nx+2*g.ng)
            ar = numpy.zeros(g.nx+2*g.ng)
            # i is looping over interfaces, so al is the right edge of the left
            # element, etc.
            for i in range(g.ilo, g.ihi+2):
                al[i] = local_a[-1, i-1]
                ar[i] = local_a[0, i]
            boundary_f = self.riemann(al, ar)
            rhs = interior_f
            rhs[0, 1:-1] += boundary_f[1:-1]
            rhs[-1, 1:-1] -= boundary_f[2:]

            # Multiply by mass matrix (inverse).
            rhs_i = 2 / g.dx * g.M_inv @ rhs

            return numpy.ravel(rhs_i, order='C')

        tmax = num_periods*self.period()

        # main evolution loop
        while self.t < tmax:
            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep()

            if self.t + dt > tmax:
                dt = tmax - self.t

            r = ode(rk_substep_scipy).set_integrator('dop853')
            r.set_initial_value(numpy.ravel(g.a), self.t)
            r.integrate(r.t+dt)
            g.a[:, :] = numpy.reshape(r.y, g.a.shape)
            g.a = self.limit()
            self.t += dt


if __name__ == "__main__":

    # Runs with limiter
    g_sin_nolimit = Grid1d(16, 1, 0, 1, 3)
    g_hat_nolimit = Grid1d(16, 1, 0, 1, 3)
    g_sin_moment = Grid1d(16, 1, 0, 1, 3)
    g_hat_moment = Grid1d(16, 1, 0, 1, 3)
    s_sin_nolimit = Simulation(g_sin_nolimit, 1, 0.5/7, limiter=None)
    s_hat_nolimit = Simulation(g_hat_nolimit, 1, 0.5/7, limiter=None)
    s_sin_moment = Simulation(g_sin_moment, 1, 0.5/7, limiter="moment")
    s_hat_moment = Simulation(g_hat_moment, 1, 0.5/7, limiter="moment")
    for s in s_sin_nolimit, s_sin_moment:
        s.init_cond("sine")
    for s in s_hat_nolimit, s_hat_moment:
        s.init_cond("tophat")
    x_sin_start, a_sin_start = g_sin_nolimit.plotting_data()
    x_hat_start, a_hat_start = g_hat_nolimit.plotting_data()
    for s in s_sin_nolimit, s_sin_moment, s_hat_nolimit, s_hat_moment:
        s.evolve()
    fig, axes = pyplot.subplots(1, 2)
    axes[0].plot(x_sin_start, a_sin_start, 'k-', label='Exact')
    axes[1].plot(x_hat_start, a_hat_start, 'k-', label='Exact')
    for i, g in enumerate((g_sin_nolimit, g_hat_nolimit)):
        axes[i].plot(*g.plotting_data(), 'b--', label='No limiting')
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel(r"$x$")
        axes[i].set_ylabel(r"$a$")
    for i, g in enumerate((g_sin_moment, g_hat_moment)):
        axes[i].plot(*g.plotting_data(), 'r:', label='Moment limiting')
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel(r"$x$")
        axes[i].set_ylabel(r"$a$")
    fig.tight_layout()
    lgd = axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#    fig.savefig('dg_limiter.pdf',
#                bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyplot.show()

    # -------------------------------------------------------------------------
    # Show the "grid" using a sine wave

    xmin = 0.0
    xmax = 1.0
    nx = 4
    ng = 1

    u = 1.0

    colors = "kbr"
    symbols = "sox"
    for m in range(1, 4):
        g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
        s = Simulation(g, u, C=0.5/(2*m+1))
        s.init_cond("sine")
        plot_x, plot_a = g.plotting_data()
        plot_x_hi, plot_a_hi = g.plotting_data_high_order()
        pyplot.plot(plot_x, plot_a, f'{colors[m-1]}{symbols[m-1]}',
                    label=fr"Nodes, $m={{{m}}}$")
        pyplot.plot(plot_x_hi, plot_a_hi, f'{colors[m-1]}:',
                    label=fr"Modes, $m={{{m}}}$")
    pyplot.xlim(0, 1)
    pyplot.vlines([0.25, 0.5, 0.75], -2, 2, linestyles='--')
    pyplot.ylim(-1, 1)
    pyplot.xlabel(r'$x$')
    pyplot.ylabel(r'$a$')
    pyplot.legend()
    pyplot.show()

    # Note that the highest m (4) doesn't converge at the expected rate -
    # probably limited by the time integrator.
    colors = "brckgy"
    symbols = "xo^<sd"
    fig, axes = pyplot.subplots(1, 3)
    ms = numpy.array(range(1, 5))
    nxs = 2**numpy.array(range(3, 9))
    errs = numpy.zeros((len(ms), len(nxs)))
    for i, m in enumerate(ms):
        for j, nx in enumerate(nxs):
            g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
            s = Simulation(g, u, C=0.5/(2*m+1))
            s.init_cond("sine")
#            s.init_cond("gaussian")
            a_init = g.a.copy()
            s.evolve(num_periods=1)
            errs[i, j] = s.grid.norm(s.grid.a - a_init)
        axes[0].loglog(nxs, errs[i, :], f'{colors[i]}{symbols[i]}')
        if m < 4:
            axes[0].plot(nxs, errs[i, -2]*(nxs[-2]/nxs)**(m+1),
                         f'{colors[i]}--')
    axes[0].set_xlabel(r'$N$')
    axes[0].set_ylabel(r'$\|$Error$\|_2$')
    axes[0].set_title('RK3')

    # To check that it's a limitation of the time integrator, we can use
    # the scipy DOPRK8 integrator
    colors = "brckgy"
    symbols = "xo^<sd"
    ms = numpy.array(range(1, 6))
    nxs = 2**numpy.array(range(3, 9))
    errs_dg = numpy.zeros((len(ms), len(nxs)))
    for i, m in enumerate(ms):
        for j, nx in enumerate(nxs):
            print(f"DOPRK8, m={m}, nx={nx}")
            g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
            s = Simulation(g, u, C=0.5/(2*m+1))
            s.init_cond("sine")
#            s.init_cond("gaussian")
            a_init = g.a.copy()
            s.evolve_scipy(num_periods=1)
            errs_dg[i, j] = s.grid.norm(s.grid.a - a_init)
        axes[1].loglog(nxs, errs_dg[i, :], f'{colors[i]}{symbols[i]}',
                       label=fr'$m={{{m}}}$')
        if m < 5:
            axes[1].plot(nxs, errs_dg[i, -2]*(nxs[-2]/nxs)**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
        else:
            axes[1].plot(nxs[:-1], errs_dg[i, -2]*(nxs[-2]/nxs[:-1])**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
    axes[1].set_xlabel(r'$N$')
    axes[1].set_ylabel(r'$\|$Error$\|_2$')
    axes[1].set_title('DOPRK8')

    # Check convergence with the limiter on
    errs_lim = numpy.zeros((len(ms), len(nxs)))
    for i, m in enumerate(ms):
        for j, nx in enumerate(nxs):
            print(f"DOPRK8, m={m}, nx={nx}")
            g = Grid1d(nx, ng, xmin=xmin, xmax=xmax, m=m)
            s = Simulation(g, u, C=0.5/(2*m+1), limiter="moment")
            s.init_cond("sine")
            a_init = g.a.copy()
            s.evolve_scipy(num_periods=1)
            errs_lim[i, j] = s.grid.norm(s.grid.a - a_init)
        axes[2].loglog(nxs, errs_lim[i, :], f'{colors[i]}{symbols[i]}',
                       label=fr'$m={{{m}}}$')
        if m < 5:
            axes[2].plot(nxs, errs_lim[i, -2]*(nxs[-2]/nxs)**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
        else:
            axes[2].plot(nxs[:-1], errs_lim[i, -2]*(nxs[-2]/nxs[:-1])**(m+1),
                         f'{colors[i]}--',
                         label=fr'$\propto (\Delta x)^{{{m+1}}}$')
    axes[2].set_xlabel(r'$N$')
    axes[2].set_ylabel(r'$\|$Error$\|_2$')
    axes[2].set_title('DOPRK8, Moment limiter')
    fig.tight_layout()
    lgd = axes[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('dg_convergence_sine.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyplot.show()
