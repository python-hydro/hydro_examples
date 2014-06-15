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

def diffuseCN(gr, phi, k, dt):
    """ diffuse phi implicitly through timestep dt, with a C-N
        temporal discretization """

    phinew = gr.scratchArray()
    
    alpha = k*dt/gr.dx**2

    # create the RHS of the matrix
    gr.fillBC()
    R = 0.5*k*dt*lap(gr, phi)
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


def lap(gr, phi):
    """ compute the Laplacian of phi """

    lapphi = gr.scratchArray()

    ib = gr.ilo
    ie = gr.ihi

    lapphi[ib:ie+1] = (phi[ib-1:ie] - 2.0*phi[ib:ie+1] + phi[ib+1:ie+2])/gr.dx**2

    return lapphi


class grid:

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


    def scratchArray(self):
        return numpy.zeros((2*self.ng+self.nx), dtype=numpy.float64)


    def phi_a(self, t, k, t0, phi1, phi2):
        """ analytic solution """

        xc = 0.5*(self.xmin + self.xmax)
        return (phi2 - phi1)*numpy.sqrt(t0/(t + t0)) * \
            numpy.exp(-0.25*(self.x-xc)**2/(k*(t + t0))) + phi1

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if not len(e) == (2*self.ng + self.nx):
            return None

        return numpy.sqrt(self.dx*numpy.sum(e[self.ilo:self.ihi+1]**2))



def evolve(nx, k, t0, phi1, phi2, C, tmax):
    """ 
    the main evolution loop.  Evolve 
  
     phi_t = k phi_{xx} 

    from t = 0 to tmax
    """

    # create the grid
    gr = grid(nx, ng=1, xmax=1.0)

    # time info
    dt = C*0.5*gr.dx**2/k
    t = 0.0

    # initialize the data
    gr.phi[:] = gr.phi_a(0.0, k, t0, phi1, phi2)

    while (t < tmax):

        gr.fillBC()

        # make sure we end right at tmax
        if (t + dt > tmax):
            dt = tmax - t

        # diffuse for dt
        phinew = diffuseCN(gr, gr.phi, k, dt)

        gr.phi[:] = phinew[:]

        t += dt

    return gr


def evolveExplicit(nx, k, t0, phi1, phi2, C, tmax):
    """ fully explicit for comparison """

    ng = 1

    # create the grid
    g = grid(nx, ng, xmax=1.0)

    # time info
    dt = C*0.5*g.dx**2/k
    t = 0.0

    # initialize the data
    g.phi[:] = g.phi_a(0.0, k, t0, phi1, phi2)

    # evolution loop
    phinew = g.scratchArray()
    
    while (t < tmax):

        # make sure we end right at tmax
        if (t + dt > tmax):
            dt = tmax - t

        # fill the boundary conditions
        g.fillBC()

        alpha = k*dt/g.dx**2

        # loop over zones
        i = g.ilo
        while (i <= g.ihi):
        
            # Lax-Wendroff
            phinew[i] = g.phi[i] + alpha*(g.phi[i+1] - 2.0*g.phi[i] + g.phi[i-1])
                
            i += 1

        # store the updated solution
        g.phi[:] = phinew[:]

        t += dt

    return g

#-----------------------------------------------------------------------------
# convergence C = 0.8


pylab.clf()

# a characteristic timescale for diffusion if L^2/k
tmax = 0.005

t0 = 1.e-4
phi1 = 1.0
phi2 = 2.0

k = 1.0

N = [32, 64, 128, 256, 512]
ng = 1

# CFL number
C = 0.8

err = []
errExpl = []

for nx in N:

    print nx

    # the present C-N discretization
    g = evolve(nx, k, t0, phi1, phi2, C, tmax)

    # compare to the explicit discretization
    gExpl = evolveExplicit(nx, k, t0, phi1, phi2, C, tmax)
    
    phi_analytic = g.phi_a(tmax, k, t0, phi1, phi2)

    err.append(g.norm(g.phi - phi_analytic))
    errExpl.append(g.norm(gExpl.phi - phi_analytic))


pylab.clf()

N = numpy.array(N, dtype=numpy.float64)
err = numpy.array(err)

print "err = ", err
print "errExpl = ", errExpl

pylab.scatter(N, err, color="r", label="C-N implicit diffusion")
pylab.scatter(N, errExpl, color="g", label="forward-diff explicit diffusion")
pylab.plot(N, err[len(N)-1]*(N[len(N)-1]/N)**2, color="k", label="$\mathcal{O}(\Delta x^2)$")

ax = pylab.gca()
ax.set_xscale('log')
ax.set_yscale('log')

pylab.xlabel(r"$N$")
pylab.ylabel(r"L2 norm of absolute error")
pylab.title("Convergence of Diffusion Methods, C = %3.2f, t = %5.2g" % (C, tmax))

pylab.ylim(1.e-6, 1.e-2)
pylab.legend(frameon=False, fontsize="small")

pylab.savefig("diffmethods-converge-0.8.png")



#-----------------------------------------------------------------------------
# convergence C = 2.0


pylab.clf()

# a characteristic timescale for diffusion if L^2/k
tmax = 0.005

t0 = 1.e-4
phi1 = 1.0
phi2 = 2.0

k = 1.0

N = [32, 64, 128, 256, 512]
ng = 1

# CFL number
C = 2.0

err = []
errExpl = []

for nx in N:

    print nx

    # the present C-N discretization
    g = evolve(nx, k, t0, phi1, phi2, C, tmax)

    phi_analytic = g.phi_a(tmax, k, t0, phi1, phi2)

    err.append(g.norm(g.phi - phi_analytic))


pylab.clf()

N = numpy.array(N, dtype=numpy.float64)
err = numpy.array(err)

print "err = ", err

pylab.scatter(N, err, color="r", label="C-N implicit diffusion")
pylab.plot(N, err[len(N)-1]*(N[len(N)-1]/N)**2, color="k", label="$\mathcal{O}(\Delta x^2)$")

ax = pylab.gca()
ax.set_xscale('log')
ax.set_yscale('log')

pylab.xlabel(r"$N$")
pylab.ylabel(r"L2 norm of absolute error")
pylab.title("Convergence of Diffusion Methods, C = %3.2f, t = %5.2g" % (C, tmax))

pylab.ylim(1.e-6, 1.e-2)
pylab.legend(frameon=False, fontsize="small")

pylab.savefig("diffmethods-converge-2.0.png")


