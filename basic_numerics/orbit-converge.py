import numpy
import pylab
import math
from orbit import *

# circular orbit
o = orbit(1.0, 0.0)   # eccentricity = 0

# orbital period
P = o.keplerPeriod()

tstep = []
errEuler = []
errEC = []
errRK2 = []
errRK4 = []

dt = 0.05
histEuler = o.intEuler(dt, P)
histEulerCromer = o.intEulerCromer(dt, P)
histRK2 = o.intRK2(dt, P)
histRK4 = o.intRK4(dt, P)

# error is final radius - initial radius.  Since we are circular, the
# initial radius is o.a, the semimajor axis
print dt, \
    abs(histEuler.finalR()-o.a), \
    abs(histEulerCromer.finalR()-o.a), \
    abs(histRK2.finalR()-o.a), \
    abs(histRK4.finalR()-o.a)

tstep.append(dt)
errEuler.append(abs(histEuler.finalR()-o.a))
errEC.append(abs(histEulerCromer.finalR()-o.a))
errRK2.append(abs(histRK2.finalR()-o.a))
errRK4.append(abs(histRK4.finalR()-o.a))


dt = 0.025
histEuler = o.intEuler(dt, P)
histEulerCromer = o.intEulerCromer(dt, P)
histRK2 = o.intRK2(dt, P)
histRK4 = o.intRK4(dt, P)

# error is final radius - initial radius
print dt, \
    abs(histEuler.finalR()-o.a), \
    abs(histEulerCromer.finalR()-o.a), \
    abs(histRK2.finalR()-o.a), \
    abs(histRK4.finalR()-o.a)

tstep.append(dt)
errEuler.append(abs(histEuler.finalR()-o.a))
errEC.append(abs(histEulerCromer.finalR()-o.a))
errRK2.append(abs(histRK2.finalR()-o.a))
errRK4.append(abs(histRK4.finalR()-o.a))


dt = 0.0125
histEuler = o.intEuler(dt, P)
histEulerCromer = o.intEulerCromer(dt, P)
histRK2 = o.intRK2(dt, P)
histRK4 = o.intRK4(dt, P)

# error is final radius - initial radius
print dt, \
    abs(histEuler.finalR()-o.a), \
    abs(histEulerCromer.finalR()-o.a), \
    abs(histRK2.finalR()-o.a), \
    abs(histRK4.finalR()-o.a)

tstep.append(dt)
errEuler.append(abs(histEuler.finalR()-o.a))
errEC.append(abs(histEulerCromer.finalR()-o.a))
errRK2.append(abs(histRK2.finalR()-o.a))
errRK4.append(abs(histRK4.finalR()-o.a))


dt = 0.00625
histEuler = o.intEuler(dt, P)
histEulerCromer = o.intEulerCromer(dt, P)
histRK2 = o.intRK2(dt, P)
histRK4 = o.intRK4(dt, P)

# error is final radius - initial radius
print dt, \
    abs(histEuler.finalR()-o.a), \
    abs(histEulerCromer.finalR()-o.a), \
    abs(histRK2.finalR()-o.a), \
    abs(histRK4.finalR()-o.a)

tstep.append(dt)
errEuler.append(abs(histEuler.finalR()-o.a))
errEC.append(abs(histEulerCromer.finalR()-o.a))
errRK2.append(abs(histRK2.finalR()-o.a))
errRK4.append(abs(histRK4.finalR()-o.a))


dt = 0.003125
histEuler = o.intEuler(dt, P)
histEulerCromer = o.intEulerCromer(dt, P)
histRK2 = o.intRK2(dt, P)
histRK4 = o.intRK4(dt, P)

# error is final radius - initial radius
print dt, \
    abs(histEuler.finalR()-o.a), \
    abs(histEulerCromer.finalR()-o.a), \
    abs(histRK2.finalR()-o.a), \
    abs(histRK4.finalR()-o.a)

tstep.append(dt)
errEuler.append(abs(histEuler.finalR()-o.a))
errEC.append(abs(histEulerCromer.finalR()-o.a))
errRK2.append(abs(histRK2.finalR()-o.a))
errRK4.append(abs(histRK4.finalR()-o.a))



pylab.scatter(numpy.array(tstep), numpy.array(errEuler), label="Euler", color="k")
pylab.plot(numpy.array(tstep), errEuler[0]*(tstep[0]/numpy.array(tstep))**-1, color="k")

pylab.scatter(numpy.array(tstep), numpy.array(errEC), label="Euler-Cromer", color="r")
pylab.plot(numpy.array(tstep), errEC[0]*(tstep[0]/numpy.array(tstep))**-1, color="r")

pylab.scatter(numpy.array(tstep), numpy.array(errRK2), label="R-K 2", color="b")
pylab.plot(numpy.array(tstep), errRK2[0]*(tstep[0]/numpy.array(tstep))**-2, color="b")

pylab.scatter(numpy.array(tstep), numpy.array(errRK4), label="R-K 4", color="g")
pylab.plot(numpy.array(tstep), errRK4[0]*(tstep[0]/numpy.array(tstep))**-4, color="g")

leg = pylab.legend(loc=2)
ltext = leg.get_texts()
pylab.setp(ltext, fontsize='small')
leg.draw_frame(0)

ax = pylab.gca()
ax.set_xscale('log')
ax.set_yscale('log')

pylab.xlabel(r"$\tau$")
pylab.ylabel("absolute error in radius after one period")

pylab.ylim(1.e-10, 10)

pylab.savefig("orbit-converge.png")

