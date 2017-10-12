import numpy
from matplotlib import pyplot
import advection
import weno_coefficients


def weno(order, q):
    """
    Do WENO reconstruction
    
    Parameters
    ----------
    
    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct
        
    Returns
    -------
    
    qL : numpy array
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order, len(q)))
    w = numpy.zeros_like(beta)
    np = len(q) - 2 * order
    epsilon = 1e-16
    for i in range(order, np+order):
        q_stencils = numpy.zeros(order)
        alpha = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l+1):
                    beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
            alpha[k] = C[k] / (epsilon + beta[k, i]**2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[i+k-l]
        w[:, i] = alpha / numpy.sum(alpha)
        qL[i] = numpy.dot(w[:, i], q_stencils)
    
    return qL


class WENOSimulation(advection.Simulation):
    
    def __init__(self, grid, u, C=0.8, weno_order=3):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.u = u   # the constant advective velocity
        self.C = C   # CFL number
        self.weno_order = weno_order


    def states(self):
        
        g = self.grid
        al = g.scratch_array()
        ar = g.scratch_array()
        
        al = weno(self.weno_order, g.a)
        ar[::-1] = weno(self.weno_order, g.a[::-1])
        return al, ar


    def rk_substep(self):
        
        g = self.grid
        g.fill_BCs()
        al, ar = self.states()
        flux = self.riemann(al, ar)
        rhs = g.scratch_array()
        rhs[g.ilo:g.ihi+1] = 1/g.dx * (flux[g.ilo-1:g.ihi] - flux[g.ilo:g.ihi+1])
 #       rhs[g.ilo:g.ihi+1] = 1/g.dx * (flux[g.ilo:g.ihi+1] - flux[g.ilo+1:g.ihi+2])
        return rhs


    def evolve(self, num_periods=1):
        """ evolve the linear advection equation using RK4 """
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

            # RK4
            # Store the data at the start of the step
            a_start = g.a.copy()
            k1 = dt * self.rk_substep()
            g.a = a_start + k1 / 2
            k2 = dt * self.rk_substep()
            g.a = a_start + k2 / 2
            k3 = dt * self.rk_substep()
            g.a = a_start + k3
            k4 = dt * self.rk_substep()
            g.a = a_start + (k1 + 2 * (k2 + k3) + k4) / 6

            self.t += dt


if __name__ == "__main__":


    #-------------------------------------------------------------------------
    # compute WENO3 case

    xmin = 0.0
    xmax = 1.0
    nx = 64
    order = 3
    ng = order+1

    g = advection.Grid1d(nx, ng, xmin=xmin, xmax=xmax)

    u = 1.0
    
    s = WENOSimulation(g, u, C=0.5, weno_order=3)

    s.init_cond("gaussian")
    ainit = s.grid.a.copy()

    s.evolve(num_periods=1)

    pyplot.plot(g.x[g.ilo:g.ihi+1], ainit[g.ilo:g.ihi+1],
             ls=":", label="exact")

    pyplot.plot(g.x[g.ilo:g.ihi+1], g.a[g.ilo:g.ihi+1],
             label="WENO3")
    
    
    #-------------------------------------------------------------------------
    # convergence test
    problem = "gaussian"

    xmin = 0.0
    xmax = 1.0
    orders = [2, 3, 4]
    N = [32, 64, 128, 256, 512]
 #   N = [32, 64, 128]

    errs = []

    u = 1.0

    for order in orders:
        ng = order+1
        errs.append([])
        for nx in N:
            print(order, nx)
            gu = advection.Grid1d(nx, ng, xmin=xmin, xmax=xmax)
            su = WENOSimulation(gu, u, C=0.5, weno_order=order)
        
            su.init_cond("gaussian")
            ainit = su.grid.a.copy()
        
            su.evolve(num_periods=1)
        
            errs[-1].append(gu.norm(gu.a - ainit))
    
    pyplot.clf()
    N = numpy.array(N, dtype=numpy.float64)
    for n_order, order in enumerate(orders):
        pyplot.scatter(N, errs[n_order], label=r"WENO, $r={}$".format(order))
    pyplot.plot(N, errs[n_order][len(N)-1]*(N[len(N)-1]/N)**4,
                color="k", label=r"$\mathcal{O}(\Delta x^4)$")

    ax = pyplot.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    pyplot.xlabel("N")
    pyplot.ylabel(r"$\| a^\mathrm{final} - a^\mathrm{init} \|_2$",
               fontsize=16)

    pyplot.legend(frameon=False)
    pyplot.savefig("weno-converge.pdf")
    #pyplot.show()
    