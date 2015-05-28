import math
import numpy
import pylab

def plot_convergence():

    data = numpy.loadtxt("converge.txt")

    nx = data[:,0]
    err = data[:,2]

    ax = pylab.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pylab.scatter(nx, err, marker="x", color="r")
    pylab.plot(nx, err[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.xlabel("number of zones")
    pylab.ylabel("L2 norm of abs error")

    f = pylab.gcf()
    f.set_size_inches(5.0,5.0)

    pylab.savefig("fft-poisson-converge.png", bbox_inches="tight")

    

if __name__== "__main__":
    plot_convergence()

