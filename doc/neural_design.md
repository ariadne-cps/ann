# Neural: design objectives and initial architecture

## Status and scope

This document records the initial design decisions for the `ariadne-cps/neural`
repository. It is a design journal and roadmap, **not** a claim that the
features below have already been implemented. The initial repository bootstrap
contains build infrastructure, a minimal public header, smoke tests, and
optional Python binding registration.

Ariadne Neural aims to provide an autonomous, portable, CPU-first C++20
framework for neural-network inference and learning. Its distinctive goal is
to preserve enough mathematical structure to support validated computation,
structural guarantees, and rigorous analysis by the rest of Ariadne.

The primary applications are dense feed-forward controllers and candidate
barrier, Lyapunov, and related certificate functions for dynamical systems.
Neural is not intended to become a competing general-purpose deep learning
ecosystem, nor to own the system-level proof workflow.

## Dependency and responsibility boundaries

Production dependencies are **Function** and **Threading**, alongside
**Configuration** for CMake. Function transitively supplies Geometry, Algebra,
Interval, Numeric, Foundation, Logging, Utility, and other lower-level Ariadne
components. Testing is pinned directly but is used **only for tests**, never
as a production target dependency. Git-submodule commits must be consistent
across all shared transitive paths.

- **Neural owns:** network architecture, parameters, activations, CPU
  inference, training, neural-specific numerical bounds, structural constraints,
  transfer learning, model serialization, and portable backend interfaces.
- **Function owns:** general Ariadne function abstractions, composition,
  differentiation and validated function models. Neural should interoperate
  with Function without forcing inference to execute through generic
  expression graphs.
- **Threading owns:** generic task scheduling, parallel loops, fixed-block
  reductions, cancellation and concurrency budgeting. Neural determines
  DNN-specific workload partitioning, buffering, and synchronization.
- **Reasoning owns:** formulation and proof of properties, neural safety
  verification, system-level controller/barrier synthesis, solver orchestration,
  counterexample-guided inductive synthesis (CEGIS), and composition with
  System, Evolution, SMT, SAT, and other reasoning components.
- **SMT owns:** general bounded-real constraint solving and logical result
  semantics. Neural has **no direct SMT dependency**.

Neural's evaluated networks, enclosures, derivatives, and structural
properties must be usable by Reasoning, preferably through existing Function
interfaces. A neural-specific adapter may be implemented in Reasoning; Neural
must not depend on Reasoning or on any solver.

No direct production dependency on Linear Programming, Nonlinear Programming,
Constraint Solving, System, or Evolution is needed at bootstrap. Some future
training algorithms may justify optional, directly pinned optimization
dependencies. Such a decision must follow a concrete use case and a DAG
consistency review, not the mere existence of a solver.

## Architectural principles

1. **One canonical model.** Network topology, layer metadata, activation
   definitions, parameter tensor shapes/layout, and parameter values have
   an unambiguous, backend-independent representation.
2. **Separate model from execution.** Approximate CPU, validated CPU, and
   future external backend evaluators must preserve agreed network semantics.
   An optimized floating-point backend is not automatically a certified
   implementation of the mathematical model.
3. **Use dedicated dense kernels.** The training engine needs efficient
   batched forward/backward propagation and gradients with respect to
   trainable parameters. Function's reverse differentiation of general
   procedures can be reused for interoperability or checking, but is not
   assumed to be an efficient DNN training implementation.
4. **No mandatory external machine-learning runtime.** The portable
   implementation relies on standard C++20 and the existing Ariadne stack.
   Optional LibTorch and MLX adapters are isolated from the core API and
   from validated execution.
5. **Preserve mathematical evidence.** Approximate results, rigorous interval
   enclosures, and proven structural properties must never be conflated.
   Failure to prove a property is not proof of its negation.
6. **Minimize premature abstractions.** The first implementation targets
   sequential dense feed-forward networks; general tensor/autograd engines,
   convolutions, recurrent models and attention are outside the initial scope.

## Candidate public model and inference concepts

The primary model is a sequential composition of affine dense layers and
per-layer activations:

```text
z[0] = x
z[k+1] = activation[k](weights[k] * z[k] + bias[k])
```

Planned activations and output transformations include identity/linear, ReLU,
tanh, sigmoid, softplus, and bounded or positive outputs. Follow-on candidates
include leaky ReLU, ELU, SiLU, GELU, softmax and residual connections.

Smoothness is part of the mathematical contract. Classical derivatives need
not exist at ReLU kinks; any nonsmooth evaluation or derivative interface
must specify its mathematical meaning. Smooth activations are especially
relevant when system-level proof obligations involve Lie derivatives.

Approximate and validated inference should use the same frozen model
parameters, with explicit conversion rules and reproducible serialization.
Validated evaluation may start with interval bound propagation, followed by
affine bounds, Taylor models, derivative enclosures, and tighter specialized
propagators. In particular, ordinary interval propagation suffers from
dependency overestimation; its enclosure guarantees must not be presented
as tightness guarantees.

## Learning, optimization and transfer learning

The initial planned native training engine includes:

- Dense-layer backpropagation, mini-batching, shuffling, reproducible
  initialization, and explicit data/gradient buffers.
- SGD, momentum, Nesterov, Adam, and AdamW; learning-rate schedules later.
- MSE, MAE, Huber, binary and multiclass cross-entropy where appropriate.
- Freeze/unfreeze, trainable parameter groups, replacement of output heads,
  fine-tuning, checkpoints, and explicit optimizer-state handling.
- CPU scheduling through Threading, preferably batching independent
  examples and computing thread-private or block-private gradients before
  reductions. Fixed-block reductions improve reproducibility across worker
  counts but do not guarantee identical floating-point results across
  platforms or correctly rounded whole sums.

Transfer learning is a first-class design requirement: saved parameter names,
shapes, activation semantics, and optimizer state should permit selective
loading and fine-tuning. A previously certified frozen model loses its
certificate's applicability after any relevant parameter or semantic change.

## Constraints and guarantees by construction

The framework must distinguish four mechanisms:

1. **Soft penalties in the loss:** encourage, but do not prove, constraint
   satisfaction.
2. **Projected updates:** can preserve a feasible parameter set if the
   projection or feasibility-restoring operation is correctly implemented.
3. **Structural parameterizations:** can enforce properties independently
   of training progress, subject to their mathematical assumptions.
4. **External certification:** Reasoning proves properties that do not
   follow solely from architecture, using rigorous numerical tools or SMT.

Candidate intrinsically enforced properties include nonnegative weights,
parameter bounds, monotonicity under admissible connectivity/activation
conditions, bounded outputs, and conservative Lipschitz bounds. Input-convex
architectures may be useful. Each guarantee must state precisely which
variables, domains, arithmetic model, and assumptions it covers.

A Lipschitz bound for a composition can be conservatively inferred from
layer operator-norm bounds and activation Lipschitz constants; the result is
valid only when those component bounds are established.

General nonlinear constraints on parameters may require specialized
optimization algorithms, but that does not imply including
`nonlinear-programming` in the core dependency graph now. Similarly,
linear constraints can be useful for later bounds and LP relaxations, while
Euclidean projection onto general linear constraints is a quadratic problem
and not automatically an LP.

A structurally admissible barrier network does **not** in general satisfy
a dynamical Lie-derivative inequality automatically. For example, constructions
of the form `B(x) = g(x) + h(x) * N(x)` may enforce boundary conditions when
`h` vanishes on a prescribed boundary, but remaining system-level conditions
must be checked by Reasoning.

## Reasoning integration and proof lifecycle

Neural should expose a frozen network through Function-compatible approximate
and validated interfaces, along with optional specialized bounds and input
derivatives. Reasoning translates these objects into proof obligations,
calls SMT and other solvers, and organizes counterexamples and subsequent
training attempts.

The existing SMT Barr3 fixture (published tanh 2-64-64-1 network with 4,417
parameters) is a useful end-to-end regression and performance baseline. Once
there is a real Neural model API, compare the model-based path with the
existing manually constructed expressions and Lie derivatives. Avoid claiming
a speedup until matching mathematical obligations are tested with controlled
budgets, machine conditions, and solver configurations.

Proof results must include the property, domain, network snapshot identity,
numerical semantics, assumptions, and solver outcome. In particular, SMT
`UNSAT`, `EPSILON_SAT`, and `UNKNOWN` are distinct; `UNKNOWN` does
not constitute certification, and epsilon-satisfiability is not an exact
counterexample without additional validation.

Once trained weights are frozen, any mathematically certified property must
remain tied to that exact model (including architecture and activation
semantics). A serialized snapshot hash is planned for this purpose.

## Planned implementation stages

1. Establish repository structure, CMake/CI, reproducible submodule pins,
   installed consumer test, C++ Testing suite, and optional Python bindings.
2. Define the canonical feed-forward model, activation semantics, serialization,
   and approximate CPU inference with independent numerical oracles.
3. Add dense backward propagation, minibatch learning, native optimizers,
   checkpointing, freezing, and transfer learning.
4. Add rigorously validated forward/derivative enclosures and constrained
   parameterizations; test mathematical inclusions rather than only
   approximate equality.
5. Expose Function-compatible representations and specialized bounds so
   Reasoning can use Neural with SMT and dynamic-system verification.
6. Add optional backend adapters only after the canonical numerical semantics,
   conversion rules and portability contract have stabilized.

## Engineering and CI policy

Follow the modular Ariadne layout: `include/neural`, `src`, `tests`,
`python/bindings`, `python/tests`, a single `tutorial/` directory holding
both `tutorial_neural.cpp` and `tutorial_neural.py`, and `doc`; provide
`include/ariadne-neural.hpp` as the umbrella header.

Follow Function's standalone build conventions: CMake 3.15+, C++20,
configuration bootstrap, test-only pinned Testing, installed bundled
headers, optional composed `pyariadne`, and `everything`, `tests`,
`bindings`, `coverage`, and `uninstall` integration where applicable.
CI should include Ubuntu, macOS, Windows, sanitizers, and coverage.

New behavior requires focused C++ and Python tests, sanitizer checks,
validated numerical oracles for rigorous paths, and documentation of
performance claims. Never weaken a semantic test just to satisfy CI or to
increase coverage. Keep runtime-optional backends out of the mandatory CI
matrix until a supported backend integration exists.
