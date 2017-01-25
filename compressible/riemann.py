# solve the Riemann problem for a gamma-law gas

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

class State(object):
    def __init__(self, p=1.0, u=0.0, rho=1.0):
         self.p = p
         self.u = u
         self.rho = rho

    def __str__(self):
        return "rho: {}; u: {}; p: {}".format(self.rho, self.u, self.p)

class RiemannProblem(object):
    def __init__(self, left_state, right_state, gamma=1.4):
        self.left = left_state
        self.right = right_state
        self.gamma = gamma

    def u_hugoniot(self, p, side):

        if side == "left":
            state = self.left
            s = 1.0
        elif side == "right":
            state = self.right
            s = -1.0

        c = np.sqrt(self.gamma*state.p/state.rho)

        if p < state.p:
            # rarefaction
            u = state.u + s*(2.0*c/(self.gamma-1.0))* \
                (1.0 - (p/state.p)**((self.gamma-1.0)/(2.0*self.gamma)))
        else:
            # shock
            beta = (self.gamma+1.0)/(self.gamma-1.0)
            u = state.u + s*(2.0*c/np.sqrt(2.0*self.gamma*(self.gamma-1.0)))* \
                (1.0 - p/state.p)/np.sqrt(1.0 + beta*p/state.p)

        return u

    def find_star_state(self, p_min=0.001, p_max=1000.0):
        # we need to root-find on
        pstar = optimize.brentq(lambda p: self.u_hugoniot(p, "left") - self.u_hugoniot(p, "right"),
                               p_min, p_max)
        ustar = self.u_hugoniot(pstar, "left")

        return pstar, ustar

    def plot_hugoniot(self, pstar, ustar, p_min = 0.0, p_max=1.5, N=200):

        p = np.linspace(p_min, p_max, num=N)
        u_left = np.zeros_like(p)
        u_right = np.zeros_like(p)

        for n in range(N):
            u_left[n] = self.u_hugoniot(p[n], "left")

        # shock for pstar > p; rarefaction for pstar < p
        ish = np.where(p < pstar)
        ir = np.where(p > pstar)

        plt.plot(p[ish], u_left[ish], c="C0", ls=":", lw=2)
        plt.plot(p[ir], u_left[ir], c="C0", ls="-", lw=2)
        plt.scatter([self.left.p], [self.left.u], marker="x", c="C0", s=40)

        for n in range(N):
            u_right[n] = self.u_hugoniot(p[n], "right")
        ish = np.where(p < pstar)
        ir = np.where(p > pstar)

        plt.plot(p[ish], u_right[ish], ls=":", lw=2, color="C1")
        plt.plot(p[ir], u_right[ir], ls="-", lw=2, color="C1")
        plt.scatter([self.right.p], [self.right.u], marker="x", c="C1", s=40)

        du = 0.025*(max(np.max(u_left), np.max(u_right)) -
                    min(np.min(u_left), np.min(u_right)))

        plt.text(self.left.p, self.left.u+du, "left",
                 horizontalalignment="center", color="C0")

        plt.text(self.right.p, self.right.u+du, "right",
                 horizontalalignment="center", color="C1")

        plt.xlim(p_min, p_max)

        plt.xlabel(r"$p$", fontsize="large")
        plt.ylabel(r"$u$", fontsize="large")

        legs = []
        legnames = []

        legs.append(plt.Line2D((0,1),(0,0), color="C0", ls=":", marker=None))
        legnames.append("shock")

        legs.append(plt.Line2D((0,1),(0,0), color="C1", ls="-", marker=None))
        legnames.append("rarefaction")

        plt.legend(legs, legnames, frameon=False, loc="best")

        plt.tight_layout()


