# plot the Hugoniot loci for a compressible Riemann problem

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import riemann


if __name__ == "__main__":

    # setup the problem -- Sod
    left = riemann.State(p=1.0, u=0.0, rho=1.0)
    right = riemann.State(p=0.1, u=0.0, rho=0.125)

    rp = riemann.RiemannProblem(left, right)

    pstar, ustar = rp.find_star_state()

    rp.plot_hugoniot(pstar, ustar)

    plt.savefig("riemann-phase.pdf")

    print(pstar, ustar)
