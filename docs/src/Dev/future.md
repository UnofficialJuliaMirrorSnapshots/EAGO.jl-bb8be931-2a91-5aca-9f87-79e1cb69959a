# Future Work

## Current Activity:
* Update CI testing.
* Specialized algorithms for relaxing ODE constrained problems and solving global and robust optimization problems.
* Extensions for nonconvex dynamic global & robust optimization.
* Provide support for mixed-integer problems.
* Update EAGO to support nonsmooth problems (requires: a nonsmooth local nlp optimizer or lexiographic AD, support for relaxations is already included).
* Performance assessment of nonlinear (differentiable) relaxations and incorporation into main EAGO routine.
* Evaluation and incorporation of implicit relaxation routines in basic solver.

## Other things on the wishlist (but not actively being worked on):
* Implement the interval constraint propagation scheme presented in Vu 2008. For improved convergences.
* A parametric bisection routine will be updated that can divide the `(X,P)` space into a series of boxes that all contain unique branches of the implicit function `p->y(p)`.
* Provide a better interface the nonconvex semi-infinite programs solvers (JuMPeR extension?).
* Add additional McCormick relaxations.
* Add handling for domain reduction of special expression forms.
