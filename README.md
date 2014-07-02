# hydro_examples
*Simple one-dimensional examples of various hydrodynamics techniques*

This is a collection of simple python codes that demonstrate some basic
techniques used in hydrodynamics codes.  All the codes are *standalone* --
there are no interdependencies.

These codes go together with the lecture notes at:

http://bender.astro.sunysb.edu/hydro_by_example/CompHydroTutorial.pdf

and with the pyro2 code:

https://github.com/zingale/pyro2



* `advection/`

  – `advection.py`: a 1-d second-order linear advection solver with a
    wide range of limiters.

* `burgers/`

  – `burgers.py`: a 1-d second-order solver for the inviscid Burgers’
    equation, with initial conditions corresponding to a shock and a
    rarefaction.

* `multigrid/`

  – `mg_converge.py`: a convergence test of the multigrid solver. A
    Poisson problem is solved at various resolutions and compared to
    the exact solution. This demonstrates second-order accuracy.

  – `mg_test.py`: a simple driver for the multigrid solver. This sets up
    and solves a Poisson problem and plots the behavior of the
    solution as a function of V-cycle number.

  – `multigrid.py`: a multigrid class for cell-centered data. This
    implements pure V-cycles. A square domain with 2 N zones (N a
    positive integer) is required.

  – `patch1d.py`: a class for 1-d cell-centered data that lives on a
    grid. This manages the data, handles boundary conditions, and
    provides routines for prolongation and restriction to other grids.

* `diffusion/`

  – `diffusion-explicit.py`: solve the constant-diffusivity diffusion
    equation explicitly. The method is first-order accurate in time,
    but second- order in space. A Gaussian profile is diffused--the
    analytic solution is also a Gaussian.

  – `diffusion-implicit.py`: solve the constant-diffusivity diffusion
    equation implicitly. Crank-Nicolson time-discretization is used,
    resulting in a second-order method. A Gaussian profile is
    diffused.

* `multiphysics/`

  – `burgersvisc.py`: solve the viscous Burgers equation. The advective
    terms are treated explicitly with a second-order accurate
    method. The diffusive term is solved using an implicit
    Crank-Nicolson discretiza- tion. The overall coupling is
    second-order accurate.

  – `diffusion-reaction.py`: solve a diffusion-reaction equation that
    propagates a diffusive reacting front (flame). A simple reaction
    term is modeled. The diffusion is solved using a second-order
    Crank-Nicolson discretization. The reactions are evolved using the
    VODE ODE solver (via SciPy). The two processes are coupled
    together using Strang-splitting to be second-order accurate in
    time.