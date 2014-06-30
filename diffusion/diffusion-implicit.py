"""
solve the diffusion equation:

 phi_t = k phi_{xx} 

with a Crank-Nicolson implicit discretization

M. Zingale (2013-04-03)
"""

import numpy
from scipy import linalg
import sys
import pylab


class Grid1d:

    def __init__(self, nx, ng=1, xmin=0.0, xmax=1.0):
        """ grid class initialization """
        
        self.nx = nx
        self.ng = ng

        self.xmin = xmin
        self.xmax = xmax

        self.dx = (xmax - xmin)/nx
        self.x = (numpy.arange(nx+2*ng) + 0.5 - ng)*self.dx + xmin

        self.ilo = ng
        self.ihi = ng+nx-1

        # storage for the solution
        self.phi = numpy.zeros((nx+2*ng), dtype=numpy.float64)

    def fillBC(self):
        """ fill the Neumann BCs """

        # Neumann BCs
        self.phi[0:self.ilo]  = self.phi[self.ilo]
        self.phi[self.ihi+1:] = self.phi[self.ihi]


    def scratch_array(self):
        return numpy.zeros((2*self.ng+self.nx), dtype=numpy.float64)


    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if not len(e) == (2*self.ng + self.nx):
            return None

        return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))



def phi_a(t, k, x, xc, t0, phi1, phi2):
    """ analytic solution for the diffusion of a Gaussian """

    return (phi2 - phi1)*numpy.sqrt(t0/(t + t0)) * \
        numpy.exp(-0.25*(x-xc)**2/(k*(t + t0))) + phi1


class Simulation:

    def __init__(self, grid, k=1.0):
        self.grid = grid
        self.t = 0.0
        self.k = k  # diffusion coefficient


    def init_cond(self, name, *args):

        if name == "gaussian":

            # initialize the data
            xc = 0.5*(self.grid.xmin + self.grid.xmax)
            t0, phi1, phi2 = args
            self.grid.phi[:] = phi_a(0.0, self.k, self.grid.x, xc, t0, phi1, phi2)


    def diffuseCN(self, dt):
        """ 
        diffuse phi implicitly through timestep dt, with a C-N
        temporal discretization 
        """

        gr = self.grid
        phi = gr.phi

        phinew = gr.scratch_array()
    
        alpha = self.k*dt/gr.dx**2

        # create the RHS of the matrix
        gr.fillBC()
        R = 0.5*self.k*dt*self.lap()
        R = R[gr.ilo:gr.ihi+1]
        R += phi[gr.ilo:gr.ihi+1] 
    
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

        # Dirichlet
        #d[0] = 1.0 + 1.5*alpha
        #d[gr.nx-1] = 1.0 + 1.5*alpha
        
        #R[0] += alpha*phi1
        #R[gr.nx-1] += alpha*phi1

        # solve
        A = numpy.matrix([u,d,l])
        phinew[gr.ilo:gr.ihi+1] = linalg.solve_banded((1,1), A, R)

        return phinew


    def lap(self):
        """ compute the Laplacian of phi """

        gr = self.grid
        phi = gr.phi

        lapphi = gr.scratch_array()

        ib = gr.ilo
        ie = gr.ihi

        lapphi[ib:ie+1] = \
            (phi[ib-1:ie] - 2.0*phi[ib:ie+1] + phi[ib+1:ie+2])/gr.dx**2

        return lapphi


    def evolve(self, C, tmax):
        """ 
        the main evolution loop.  Evolve 
        
        phi_t = k phi_{xx} 
        
        from t = 0 to tmax
        """

        gr = self.grid

        # time info
        dt = C*0.5*gr.dx**2/self.k

        while self.t < tmax:

            gr.fillBC()

            # make sure we end right at tmax
            if (self.t + dt > tmax):
                dt = tmax - self.t

            # diffuse for dt
            phinew = self.diffuseCN(dt)

            gr.phi[:] = phinew[:]

            self.t += dt




#-----------------------------------------------------------------------------
# Convergence of a Gaussian

# a characteristic timescale for diffusion is L^2/k
tmax = 0.005

t0 = 1.e-4
phi1 = 1.0
phi2 = 2.0

k = 1.0

N = [32, 64, 128, 256, 512]


# CFL number
C = 0.8

err = []

for nx in N:

    # the present C-N discretization
    g = Grid1d(nx, ng=1)
    s = Simulation(g, k=k)
    s.init_cond("gaussian", t0, phi1, phi2)
    s.evolve(C, tmax)
    
    xc = 0.5*(g.xmin + g.xmax)
    phi_analytic = phi_a(tmax, k, g.x, xc, t0, phi1, phi2)

    err.append(g.norm(g.phi - phi_analytic))

    # pylab.clf()
    # pylab.plot(g.x[g.ilo:g.ihi+1], phi_analytic[g.ilo:g.ihi+1], color="0.5")
    # pylab.scatter(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], color="r", marker="x")

    # pylab.xlim(g.xmin, g.xmax)
    # pylab.title("N = {}".format(nx))
    # pylab.xlabel("x")
    # pylab.ylabel(r"$\phi$")

    # pylab.savefig("phi-implicit-N{}.png".format(nx))


pylab.clf()

N = numpy.array(N, dtype=numpy.float64)
err = numpy.array(err)

pylab.scatter(N, err, color="r", label="C-N implicit diffusion")
pylab.loglog(N, err[len(N)-1]*(N[len(N)-1]/N)**2, color="k", label="$\mathcal{O}(\Delta x^2)$")


pylab.xlabel(r"$N$")
pylab.ylabel(r"L2 norm of absolute error")
pylab.title("Convergence of C-N Implicit Diffusion, C = %3.2f, t = %5.2g" % (C, tmax))

pylab.ylim(1.e-6, 1.e-2)
pylab.legend(frameon=False, fontsize="small")

pylab.savefig("diffimplicit-converge-{}.png".format(C))


#-----------------------------------------------------------------------------
# solution at multiple times

# diffusion coefficient
k = 1.0

# reference time
t0 = 1.e-4

# state coeffs
phi1 = 1.0
phi2 = 2.0


nx = 128

# a characteristic timescale for diffusion is 0.5*dx**2/k
dt = 0.5/(k*nx**2)
tmax = 100*dt

# analytic on a fine grid
nx_analytic = 512

CFL = [0.8, 8.0]

for alpha in CFL:

    pylab.clf()

    ntimes = 5
    tend = tmax/2.0**(ntimes-1)

    c = ["0.5", "r", "g", "b", "k"]

    while tend <= tmax:

        g = Grid1d(nx, ng=2)
        s = Simulation(g, k=k)
        s.init_cond("gaussian", t0, phi1, phi2)
        s.evolve(alpha, tend)
        
        ga = Grid1d(nx_analytic, ng=2)
        xc = 0.5*(ga.xmin + ga.xmax)
        phi_analytic = phi_a(tend, k, ga.x, xc, t0, phi1, phi2)
        
        color = c.pop()
        pylab.plot(g.x[g.ilo:g.ihi+1], g.phi[g.ilo:g.ihi+1], 
                   "x", color=color, label="$t = %g$ s" % (tend))
        pylab.plot(ga.x[ga.ilo:ga.ihi+1], phi_analytic[ga.ilo:ga.ihi+1], 
                   color=color, ls=":")

        tend = 2.0*tend


    pylab.xlim(0.35,0.65)
    pylab.ylim(0.95,1.7)

    pylab.legend(frameon=False, fontsize="small")

    pylab.xlabel("$x$")
    pylab.ylabel(r"$\phi$")
    pylab.title(r"implicit diffusion, N = %d, $\alpha$ = %3.2f" % (nx, alpha))

    f = pylab.gcf()
    f.set_size_inches(8.0, 6.0)


    pylab.savefig("diff-implicit-{}-CFL_{}.png".format(nx, alpha))
    pylab.savefig("diff-implicit-{}-CFL_{}.eps".format(nx, alpha), bbox_inches="tight")


