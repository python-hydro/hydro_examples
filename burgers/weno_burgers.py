import numpy
from matplotlib import pyplot
import burgers
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


def weno_M(order, q):
    """
    Do WENOM reconstruction following Gerolymos equation (18)
    
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
        alpha_JS = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l+1):
                    beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
            alpha_JS[k] = C[k] / (epsilon + beta[k, i]**2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[i+k-l]
        w_JS = alpha_JS / numpy.sum(alpha_JS)
        alpha = w_JS * (C + C**2 - 3 * C * w_JS + w_JS**2) / \
                       (C**2 + w_JS * (1 - 2 * C))
        w[:, i] = alpha / numpy.sum(alpha)
        qL[i] = numpy.dot(w[:, i], q_stencils)
    
    return qL


class WENOSimulation(burgers.Simulation):
    
    def __init__(self, grid, C=0.5, weno_order=3):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.weno_order = weno_order


    def burgers_flux(self, q):
        return 0.5*q**2


    def rk_substep(self):
        
        g = self.grid
        g.fill_BCs()
        f = self.burgers_flux(g.u)
        alpha = numpy.max(abs(g.u))
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        fpr[1:] = weno(self.weno_order, fp[:-1])
        fml[-1::-1] = weno(self.weno_order, fm[-1::-1])
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
#        for i in range(len(flux)):
#            print(i,g.u[i],fpl[i],fmr[i],flux[i])
        rhs = g.scratch_array()
#        rhs[g.ilo:g.ihi+1] = 1/g.dx * (flux[g.ilo-1:g.ihi] - flux[g.ilo:g.ihi+1])
#        rhs[g.ilo:g.ihi+1] = 1/g.dx * (flux[g.ilo:g.ihi+1] - flux[g.ilo+1:g.ihi+2])
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs


    def evolve(self, tmax):
        """ evolve the linear advection equation using RK4 """
        self.t = 0.0
        g = self.grid

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            dt = self.timestep(self.C)

            if self.t + dt > tmax:
                dt = tmax - self.t

            # RK4
            # Store the data at the start of the step
            u_start = g.u.copy()
            k1 = dt * self.rk_substep()
            g.u = u_start + k1 / 2
            k2 = dt * self.rk_substep()
            g.u = u_start + k2 / 2
            k3 = dt * self.rk_substep()
            g.u = u_start + k3
            k4 = dt * self.rk_substep()
            g.u = u_start + (k1 + 2 * (k2 + k3) + k4) / 6

            self.t += dt



if __name__ == "__main__":

    #-----------------------------------------------------------------------------
    # sine
    
    xmin = 0.0
    xmax = 1.0
    nx = 256
    order = 3
    ng = order+1
    g = burgers.Grid1d(nx, ng, bc="periodic")
    
    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin)/1.0
    
    C = 0.5
    
    pyplot.clf()
    
    s = WENOSimulation(g, C, order)
    
    for i in range(0, 10):
        tend = (i+1)*0.02*tmax
        s.init_cond("sine")
    
        uinit = s.grid.u.copy()
    
        s.evolve(tend)
    
        c = 1.0 - (0.1 + i*0.1)
        g = s.grid
        pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=str(c))
    
    
    g = s.grid
    pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="0.9", zorder=-1)
    
    pyplot.xlabel("$x$")
    pyplot.ylabel("$u$")
    pyplot.savefig("weno-burger-sine.pdf")
    
    
    #-----------------------------------------------------------------------------
    # rarefaction
    
    xmin = 0.0
    xmax = 1.0
    nx = 256
    order = 3
    ng = order+1
    g = burgers.Grid1d(nx, ng, bc="outflow")
    
    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin)/1.0
    
    C = 0.5
    
    pyplot.clf()
    
    s = WENOSimulation(g, C, order)
    
    for i in range(0, 10):
        tend = (i+1)*0.02*tmax
    
        s.init_cond("rarefaction")
    
        uinit = s.grid.u.copy()
    
        s.evolve(tend)
    
        c = 1.0 - (0.1 + i*0.1)
        pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=str(c))
    
    
    pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="0.9", zorder=-1)
    
    pyplot.xlabel("$x$")
    pyplot.ylabel("$u$")
    
    pyplot.savefig("weno-burger-rarefaction.pdf")
