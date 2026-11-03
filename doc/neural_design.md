# Neural: design objectives and initial architecture

## Status and scope

This document records the initial design decisions for the `ariadne-cps/neural`
repository. It is a design journal and roadmap, **not** a claim that the
features below have already been implemented. The initial repository bootstrap established build infrastructure and smoke
bindings. M1 implements canonical network topology and frozen binary64 snapshots;
M2 implements approximate native CPU inference, and M3 implements native CPU
reverse-mode derivatives. Training, persistence, and validation remain planned.

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
2. **Separate model from execution.** The canonical network and its parameter
   snapshot are independent from backend-owned tensors, execution graphs,
   devices, and optimizer state. Approximate CPU, validated Ariadne, LibTorch,
   and MLX must implement compatible model semantics; they need not use the
   same algorithms, memory layout, derivatives, or numerical precision.
   An optimized floating-point backend is not a certified implementation.
3. **Use dedicated dense kernels where appropriate.** The native CPU engine
   needs efficient batched forward/backward propagation and gradients with
   respect to trainable parameters. External backends should use their own
   tensor operations, autodiff and optimizers rather than emulate the native
   backpropagation engine. Function's reverse differentiation can help with
   interoperability or checking, not as the default DNN training engine.
4. **No mandatory external machine-learning runtime.** The portable
   implementation relies on standard C++20 and the existing Ariadne stack.
   LibTorch and MLX are optional, independently compiled C++ integrations,
   isolated from core public headers and validated execution. The backend
   abstraction must be defined before implementing native CPU inference.
5. **Preserve mathematical evidence.** Approximate results, rigorous interval
   enclosures, and proven structural properties must never be conflated.
   Failure to prove a property is not proof of its negation.
6. **Minimize premature abstractions.** The first implementation targets
   sequential dense feed-forward networks; general tensor/autograd engines,
   convolutions, recurrent models and attention are outside the initial scope.

## Multi-backend model and execution architecture

**Design decision (2026-11-03; specified, not yet implemented):** The
general neural-network abstraction is a backend-independent *model definition
and parameter snapshot*, with several materially different execution
implementations. A common public facade is desirable; a common internal tensor
engine, universal autodiff tape, or base class inherited by all backend
tensors is not. In particular, native CPU implementation details must not
become assumptions of the PyTorch and MLX integrations.

### Three conceptual layers

1. **Model and parameter semantics (backend-neutral):** immutable topology
   and activation descriptors; dimension and parameter-shape validation;
   layer and parameter identities; semantic restrictions and structural
   constraints; explicit weight and bias order; parameter-value format;
   serialization and frozen model identity. The canonical model contains
   no `torch::Tensor`, `mlx::core::array`, device handle, task pool, or
   execution graph.
2. **Backend-specific execution and learning:** a lightweight C++ facade
   dispatches a *whole evaluation, derivative request, or training step* to
   a backend implementation or execution session. The native CPU backend
   can use Ariadne arrays and specialized loops; LibTorch uses the PyTorch
   C++ frontend; MLX uses the MLX C++ array/transformation API; validated
   evaluation uses Ariadne's rigorous arithmetic. Do not insert virtual
   calls or canonical-tensor conversions into individual neuron operations.
3. **Backend-independent orchestration:** inference and training requests,
   loss and optimizer *descriptors*, checkpoint policies, freeze masks, and
   test reporting refer to model semantics. Their implementation is delegated
   to the selected backend, which may reject unsupported capabilities.
   A universal `Optimizer` instance or identical internal optimizer state
   across runtimes is explicitly **not** required.

Illustrative control flow (not a committed C++ API):

```text
                CanonicalNetwork + ParameterSnapshot
                                   |
                    Backend selection / capabilities
                                   |
        +--------------------------+------------------------+
        |             |            |                       |
    Native CPU    Ariadne validated    LibTorch C++       MLX C++
        |             |            |                       |
    inference     interval bounds    inference/train    inference/train
    training      derivative bounds  native autograd     native autodiff
    backprop      no default train   native optimizer    native execution
        |             |            |                       |
        +-------------+------------+-----------------------+
                      |
          explicit parameter export / snapshot
                      |
            Function-compatible frozen view
                      |
          Reasoning verification (outside Neural)
```

Each implementation may privately compile or cache an efficient model
representation. Cache keys must include model topology, parameter snapshot
identity, numeric type, device, and every option that affects execution
semantics. Training updates a backend's mutable working parameters; exporting
a frozen snapshot is an explicit operation. It must not silently mutate a
previously exported snapshot or invalidate an already established certificate
without recording that change.

### Planned backend capabilities

The following table states intended development targets, **not functionality
present in the repository at this date**.

| Backend | Approximate inference | Training/derivatives | Rigorous enclosures | Dependency |
| --- | --- | --- | --- |
| 2026-11-03 | Layout and README correction | Implemented; CI pending | Relocated the CPU inference public header to `include/neural/inference.hpp` and source to `src/inference.cpp` (no nested folders). Updated CMake and include directives. README now lists only actual capabilities and dependencies and adds the standard five workflow badges and Codecov badge. No inference behavior changed; test/build CI still required. | --- |
| Native CPU | Yes, reference and optimized implementations | Manual backpropagation and native optimizers | No, unless executed through a validated evaluator | Existing Ariadne dependencies only |
| Ariadne validated | Rigorous input/output evaluation | Validated input derivative enclosures; no default optimizer | Yes, with stated arithmetic/domain hypotheses | Existing Ariadne numerical stack |
| LibTorch | Yes | PyTorch C++ autograd and optimizers, subject to capability checks | No | Optional LibTorch integration |
| MLX | Yes | MLX C++ differentiation and framework-native training steps, subject to capability checks | No | Optional Apple/MLX integration |

Capability negotiation must be explicit and feature-granular: operations,
activations, data types, device, batch support, gradient mode, training,
serialization/conversion and validated computation. Unsupported combinations
must fail clearly or require an *explicit* fallback; they must never silently
lose requested validation guarantees. Backend support must be confirmed by
builds and tests against selected library versions, not guessed from the
existence of a Python API.

The first public facade should make backend selection and capabilities
observable but keep the exact class names provisional until M1 tests its
requirements. Internal implementations may use compile-time templates,
type-erasure or PIMPL as convenient. Do **not** force a virtual per-layer
or per-scalar interface, a generic high-level tensor language, or a dependency
on LibTorch/MLX just to build a native-only consumer.

### M1 implemented contracts and explicit limitations

The M1 C++ public API is in `include/neural/network.hpp` and
`include/neural/backend.hpp`; implementations are confined to
`src/network.cpp` and `src/backend.cpp`. Declarations are documented with
Doxygen, including exceptions, mathematical indexing and implementation
limits. The existing `include/ariadne-neural.hpp` umbrella includes both.

- `NeuralNetwork(input_dimension, layers)` requires a positive input
  dimension and at least one dense layer, with a positive output dimension
  and a recognized elementwise activation at each layer. A zero-layer or
  zero-dimensional network is unsupported; this is a deliberate initial
  restriction rather than silently modeling an identity network.
- Every layer's `weights_offset` denotes a row-major logical
  `[outputs, inputs]` array, followed immediately by its bias vector
  starting at `biases_offset`. All counts and offsets are checked for
  `size_t` overflow *before* allocating parameter arrays. Activation
  evaluation itself is not yet implemented.
- `NeuralParameterSnapshot` owns a frozen copy or moved instance of
  its topology plus an exact-length vector of finite `double` values.
  Copying or exporting it produces independent values. In this phase
  binary64 values are the canonical approximation parameter format;
  future exact/validated parameter representations must be separately
  specified, not silently converted. Signed zero is preserved.
- `NeuralExecutionSession` owns its imported frozen snapshot.
  Only `NativeCpu` is currently marked available; only
  `ImportSnapshot` is marked supported. Selecting AriadneValidated,
  LibTorch or MLX raises an error rather than silently falling back.
  Calling `require(Inference)`, `require(Training)` or
  `require(ValidatedBounds)` on the CPU session raises an error. These
  capability results must be updated **only alongside actual backend
  implementation and its tests**.
- Python bindings expose the same metadata and snapshots, with exported
  parameter vectors returned as Python-owned copies, not borrowed
  mutable storage. The tutorial examples illustrate construction and
  capability inspection, **not** forward inference.

The native session currently stores the snapshot directly, with no
backend-specific graph, execution cache or device state. An execution
request interface may be added in M2, and a mutable training workspace in
M4; neither should alter the frozen snapshot semantics.

**Validation commands for this increment:**

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DWARNINGS_AS_ERRORS=ON
cmake --build build --parallel --target tests
ctest --test-dir build --output-on-failure
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target ariadne-neural bindings
```

Test files added in M1: `tests/test_network.cpp`,
`tests/test_backend.cpp`, and `python/tests/test_network.py`.
These cover indexing, validation, overflow, finite parameters, copies,
rejection of unavailable backends, and capability declarations.
The existing Python CTest command now discovers the full
`python/tests/` directory.

At authorship, compiling the full Ariadne stack locally was not possible
because the execution environment could not resolve github.com to fetch
Git submodules. CI is the independent compilation and runtime gate.
The existing GCC coverage failure due to an unused `include/*` pattern
is unchanged; as previously agreed, this phase does not suppress it.

### Representation, conversion and numerical contracts

Define the canonical dense affine operator as
`y[o] = bias[o] + sum_i weight[o,i] * input[i]`, with
`weight` shaped `[out_features, in_features]`. Define batched inputs
as logical `[batch, features]`. These are **logical indexing contracts**,
not a requirement that every runtime stores tensors identically.

- Specify element type/precision, array orientation, parameter ordering,
  strides, activation conventions, and treatment of non-finite values.
  Each adapter must map the canonical format to its native representation.
- Store canonical checkpoints using a versioned, backend-independent
  representation that can preserve the intended numeric values/bit patterns.
  External framework checkpoints and optimizer states may be supported as
  separate, backend-specific artifacts; they do not replace the canonical
  interchange format.
- Parameter transfer is explicit. Materialize asynchronous/lazy data and
  synchronize devices as needed; report precision conversions and reject
  unsupported dtypes rather than silently losing mathematical information.
  A conversion that changes parameters produces a distinct snapshot.
- Checkpoints used for validation identify the *frozen* model and semantic
  activation behavior, not an opaque external execution graph. Rigorous
  validation starts from explicitly represented parameter values and
  rounding assumptions, **not** from an unvalidated GPU output.
- Mathematical equivalence and bounded approximate numerical error are
  the interoperability goal, not bitwise equality across CPU, CUDA, Metal
  or differing execution orders. Optimizer trajectories may diverge.
- Distinguish input derivatives, parameter gradients, reverse-mode
  vector-Jacobian products and full Jacobians. The backend advertises
  which are supported. ReLU and other nonsmooth activations require
  documented derivative conventions.

### Model exchange and persistence format decision (2026-11-03)

**Status: design agreed, not implemented.** There are two separate
interfaces with different purposes:

- **Portability: ONNX.** Import/export supported model graphs using the
  public ONNX specification. Initially support the exact sequential dense
  topology and activation set actually implemented by Neural, with explicit
  weight orientation, bias semantics, operator/opset handling, and data type
  conversion. Unsupported graph constructs or semantically incompatible
  operators must fail with an informative error, never silently approximate
  or discard parts of a model. No mandatory ONNX runtime dependency is
  introduced into the Neural core.
- **Canonical storage: JSON + Safetensors.** Persist versioned Neural
  topology, activations, declared numeric representation, tensor
  names/shapes, semantic conventions, and compatibility metadata in a
  documented JSON manifest. Store weights and biases in an open
  Safetensors file, not an executable pickle or a new ad hoc binary
  container. The manifest schema is an openly documented, versioned
  *Ariadne-specific* contract; Safetensors by itself does not describe
  a complete neural-network architecture.

The canonical representation remains
`NeuralNetwork + NeuralParameterSnapshot`; neither ONNX nodes nor
Safetensors object types enter the mathematical model or execution APIs.
For the existing binary64 snapshots, internal round trips must preserve
each parameter's binary representation, including signed zero, and
reject mismatched dimensions, unknown versions, missing/duplicate
tensors, invalid types and malformed metadata. Import/export must
apply resource limits when parsing untrusted files. Model identity
and future certificate references should be bound to a well-defined
canonical topology, activation semantics and exact parameter values;
the precise schema version and digest construction remain to be specified
before implementation.

ONNX conversion has a distinct **semantic equivalence** obligation:
mapping `Gemm`, transpositions, activations and parameter types to Neural
must be explicit, and ONNX round trips need not preserve arbitrary graph
layout, execution order or runtime-specific numerical rounding. Importing
ONNX never, by itself, grants a validated-inference guarantee. Any
conversion of parameters which changes their values produces a different
canonical snapshot and must be reported; binary32 inputs are not silently
relabelled native binary64. Direct loaders for native PyTorch, TensorFlow
or Keras checkpoints are **not** part of this decision: framework-specific
conversion can be performed externally.

**Planned order:** formalize the manifest schema and canonical exact
round-trip tests as part of M5, implement JSON + Safetensors checkpoint
support, then add ONNX import/export for the supported dense subset
with independent Python/C++ compatibility fixtures. The choice of
transport/serialization libraries is deferred until a dependency and
portability review. These statements describe planned adapters, not
implemented features.

### Build and test separation

The current layout keeps public headers directly in `include/neural/` and
implementation sources directly in `src/`, with no nested directories in either.
The CPU inference header is `include/neural/inference.hpp`; its compiled
implementation is `src/inference.cpp`.

The core CMake targets and default CI must not require external runtime
headers, libraries or Python packages. Optional integrations should be
enabled explicitly, with independent dependency discovery, version
compatibility checks, and CI on suitable machines. MLX integration is
platform-constrained; it cannot be made a mandatory Windows/Linux test
dependency. Do not assume the higher-level MLX Python `mlx.nn` API has
a class-for-class equivalent in C++; use the C++ array and differentiation
capabilities as the integration baseline.

### Compatibility test protocol

For each external backend, test a common fixture set that includes an
affine network, a multilayer smooth network, a ReLU network (excluding kink
points for classical gradient comparison), and at least one minibatch
training step.

1. Export a canonical snapshot; import it into the backend; compare
   inference against a separate analytic or reference CPU oracle.
2. Compare input derivatives and parameter gradients against independently
   established formulas away from nondifferentiable points.
3. For training, compare one update with an independently calculated step
   **only** when optimizer definitions, loss reduction and numerical
   precision match. Record differences in weight decay, epsilon placement,
   scheduling, gradient accumulation and RNG semantics.
4. Export a trained snapshot; ensure the validated native backend can
   consume its explicitly typed parameters, with conversion errors
   accounted for. Do not regard external-backend predictions as validated.
5. Verify capability rejection, precision and device errors, model identity,
   frozen parameters and copy/synchronization semantics.

A small optional LibTorch/MLX integration spike should be considered after
M2 (and an autodiff/training spike after M3/M4) to reveal API incompatibilities
early. Full optional backend packaging and performance tuning remain in M9;
the abstraction itself is a required deliverable of M1.

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

## Implementation roadmap and delivery criteria

Development is incremental. Each milestone below should produce a reviewable
vertical slice: real C++ implementation, focused semantic tests, matching Python
coverage where a public binding is available, and a usable example in the
shared `tutorial/` directory. No API should be advertised as implemented
solely because it appears in this roadmap.

The milestones are **ordered by technical dependency**, not by calendar date.
Do not start the next milestone with known baseline CI regressions unless
those failures are understood and explicitly recorded. Cross-platform
reproducibility is evaluated as a documented contract, not inferred from
deterministic scheduling alone.

### M0 — Stabilize the repository baseline

**Deliverables:** Diagnose and fix the Linux/GCC coverage job; establish
stable, reproducible Debug and Release builds, tests, installation, and
C++/Python tutorial checks. Verify dependency gitlink consistency and the
five existing CI workflows (Ubuntu, macOS, Windows, sanitizers, coverage).

**Acceptance:** All required CI jobs pass at one revision; the fix does not
weaken test assertions, suppress relevant instrumentation, or mask real
failures. Existing bootstrap APIs and the shared `tutorial/` layout remain
unchanged.

**Initial observation (2026-11-03):** On earlier completed runs, the
Ubuntu, macOS, Windows, and sanitizer workflows passed. The coverage
workflow failed in its Ubuntu/GCC job during the build/test/coverage
generation step, while its macOS job passed. Root cause has **not yet**
been established. Treat this as an open investigation, not as a diagnosed
code defect.

### M1 — Canonical model, parameter invariants and backend contracts

**Deliverables:** Define the sequential dense feed-forward architecture,
activation descriptors, logical parameter shapes and order, weights, biases,
immutable snapshot semantics and explicit dimension validation. Specify
whether numeric representation is part of the model or its executor, and
distinguish a mathematical model from a concrete floating-point execution.
Define backend selection, capabilities, request/response semantics, parameter
import/export and execution-session ownership **before** writing the native
CPU evaluator. Keep backend-private tensor, graph, device and optimizer types
out of public headers.

**Acceptance:** Construct and inspect networks from C++ and Python with
identical layer and parameter metadata. Reject invalid shapes and
nonrepresentable configurations without silent truncation. Define empty
network and zero-dimensional cases explicitly rather than accidentally.
Exercise a minimal native backend contract and unsupported-capability case;
show that the same canonical network and frozen snapshot can be handed to a
future external backend without introducing its headers into the core.
No general-purpose tensor/autograd engine.

**M1 implementation status:** topology, frozen binary64 parameter vectors,
backend enumeration, capabilities and snapshot-import sessions are implemented
with C++ and Python APIs and focused tests. The current NativeCpu backend can
only import a snapshot; it has no forward pass or training. Backend-specific
device configuration, general-purpose execution requests, persistent
serialization and non-binary64 parameters are explicitly deferred.


### M2 — Reference CPU inference

**Deliverables:** Implement native CPU as the first concrete execution
backend behind the M1 contract, with one-sample and batched dense forward
evaluation using dedicated kernels and stable elementary activations: identity,
tanh, sigmoid, ReLU, and softplus. Add a sequential reference evaluator
and define non-finite input/parameter policies, dimension checks,
storage orientation, and numerical comparison rules.

**Acceptance:** Independently calculated small-network examples agree
within explicitly justified approximate tolerances; identity and
single-layer affine cases agree with analytic results; consistent
inference is exposed through C++ and Python and demonstrated in both
`tutorial_neural.cpp` and `tutorial_neural.py`.

**M2 implementation (2026-11-03):** Native CPU now implements synchronous
`NeuralExecutionSession::infer(input)` and `infer_batch(inputs)`, mirrored in
Python. Inputs are logical [batch, input_dimension] rows and results are
[batch, output_dimension] rows; an empty batch produces an empty result.
Every row is dimension- and finiteness-validated before the batch evaluates.
The dedicated `src/inference.cpp` evaluates canonical output-major
weights, accumulating from the bias with binary64 `std::fma` in feature
order. Identity, ReLU, tanh, sigmoid (sign-conditional exponential), and
softplus (stable `log1p` formulation) are supported. This is approximate
binary64 inference, not validated arithmetic; cross-platform bit-for-bit
equivalence is not claimed. Nonfinite affine intermediate/output values
raise `std::overflow_error` instead of silently propagating NaNs/Infs.
Frozen snapshots remain immutable; derivatives, training and all
non-NativeCpu backends remain unsupported. Focused C++ and Python tests
cover two-layer ReLU, batch, empty batch, shape errors, nonfinite inputs,
overflow, activation extremes and frozen-state invariants, and both
tutorials demonstrate inference. Validation gate: run CMake builds and
all CI workflows at the new revision, particularly Windows/MSVC.
The earlier Ubuntu/GCC coverage issue was green at `3bde4b7`, while
Windows/MSVC builds remained red. Neither is claimed repaired by
M2 without a new green run. Next steps: address M0 Windows, verify
M2 on platforms, then proceed to M3 derivatives.

### M3 — Derivatives and backpropagation

**Deliverables:** Implement native CPU layerwise reverse-mode derivatives
with respect to inputs and to trainable weights/biases, plus per-batch gradient
accumulation. Backend-native autodiff remains an implementation alternative. Differentiate losses from activations, parameter storage,
and optimizer state. Define the ReLU kink policy for approximate training;
never present this convention as a classical derivative valid at zero.

**Acceptance:** Analytical cases and independent centered finite
differences agree away from nonsmooth points; gradient shapes, scaling
(sum versus mean), bias accumulation, and input Jacobians are tested.
Do not use finite differences themselves as a formal guarantee.

**M3 native CPU implementation (2026-11-03):** The existing frozen
`NeuralExecutionSession` exposes a scalar-output vector-Jacobian product
`differentiate(input, output_cotangent)` returning the input gradient and
flattened canonical parameter gradient; `input_jacobian(input)` returns
`[output_dimension, input_dimension]` rows; `differentiate_batch(inputs,
output_cotangents, reduction)` returns per-row input gradients and a
sum/mean-reduced parameter gradient. The output cotangent specifies the
per-sample scalar objective as its dot product with the network output;
no loss, optimizer or mutable training workspace is assumed. For mean
reduction **both** parameter gradients and per-sample input gradients are
divided by the batch size. Empty sum batches yield a zero parameter
gradient and no input rows; empty mean batches are rejected.
The CPU implementation shares M2's forward activation and affine kernels
and caches per-layer preactivations/outputs only when reverse mode is
requested. Backpropagation uses reverse layer order, transposed linear
maps, and outer products for weight gradients. The ReLU computational
derivative at exactly zero is **0**, a chosen subgradient convention,
not a mathematical differentiability claim. Sigmoid and softplus use
stable exponentials; tanh uses `1 - tanh(z)^2`.
Parameters and input/cotangent shapes must match the canonical snapshot.
Non-finite inputs/cotangents are rejected; non-finite products or
accumulations throw instead of silently exporting invalid gradients.
FMA order, overflow behavior, and floating-point results are
approximate and not certificates. The `InputDerivatives` and
`ParameterGradients` capability flags become true for NativeCpu only;
Training and ValidatedBounds remain unsupported. C++ tests cover
analytic affine Jacobians/VJPs, independent centered finite-difference
checks on a multilayer smooth network, kink policy, activation extremes,
batch sums/means, validation/overflow, and snapshot immutability.
Matching Python tests and tutorials exercise the public binding.
The Python CTest test registration is corrected to discover the
entire Python tests directory; previously it only ran `test_neural.py`.
The full CI gate for this increment remains pending; do not infer success
from prior green M2 runs.

### M4 — Native training and optimizers

**Deliverables:** Add backend-neutral training requests plus native-CPU
datasets/batches, reproducible initialization and shuffling, MSE and at least
one classification loss, SGD (with optional momentum), and AdamW; extend with Adam, Huber, and other losses
when semantics and tests are established. Introduce Threading-based
mini-batch execution after a correct sequential implementation exists.

**Acceptance:** Convergence is exercised on deterministic small regression
and classification tasks; optimizer update steps match independent
calculations; learning-rate, loss reduction, random seeds, exception
behavior, and deterministic fixed-block reduction policies are tested.
Thread counts must not change the mathematical API contract.

The global capability query `neural_backend_supports(NativeCpu, Training)`
is backend-wide and returns true, but the frozen
`NeuralExecutionSession::supports(Training)` returns false and
`require(Training)` fails: actual training requests are supported only
through the separate mutable `NeuralTrainingSession`. This distinction
is now covered in C++ and Python tests (M4 contract clarification).

**M4 native CPU implementation (2026-11-03):** Implemented sequential
mini-batch training as a separate `NeuralTrainingSession` that owns its
working parameter snapshot, first/second optimizer moments, and completed
step count. `NeuralExecutionSession` remains a frozen inference/derivatives
view. Training imports and exports independent immutable
`NeuralParameterSnapshot` instances; failed validation or arithmetic does
not commit parameters, moments, or step count. No dataset pointers or
mutable references are exported. The `NeuralOperation::Training` capability
now reflects the native CPU training implementation; other backends still
advertise no training.

`MeanSquaredError` evaluates `0.5 * sum_o (y_o - t_o)^2` per sample.
`BinaryCrossEntropyWithLogits` uses a numerically stable softplus
decomposition with independent binary targets in [0,1], requires
an Identity output activation, and has output cotangent
`sigmoid(logit) - target`. All batches are nonempty, have matching
input/target row counts and fixed input/output dimensions, and reject
nonfinite inputs and targets. Losses and gradients are **mean-reduced over
samples only**, not over output components. The existing M3 VJP provides
the layerwise parameter gradients. CPU accumulation is deliberately
sequential; no Threading-based parallel reductions are yet added.

SGD uses a classical momentum buffer `v = momentum*v + (g + wd*p)`,
followed by `p -= lr*v`; its weight decay is coupled L2, including
biases. AdamW uses standard first and second moments with power-based
bias correction and epsilon **outside** the square root:
`p = p*(1 - lr*wd) - lr*mhat/(sqrt(vhat) + epsilon)`; its decay is
decoupled and also applies to all parameters, including biases.
All optimizer hyperparameters are checked; arithmetic overflow fails
explicitly. `neural_initialize_parameters` generates Xavier-uniform
weights with zero biases from a fixed SplitMix64 generator; and
`neural_shuffled_indices` provides a reproducible Fisher-Yates order
with rejection sampling to avoid modulo bias. These are deterministic
given the same C++ implementation and seed, not a guarantee of
cross-platform bit-identical transcendental results.

Added focused C++ `tests/test_training.cpp` with analytic optimizer
step oracles, batch mean scaling, small regression/classification
convergence, initialization/shuffling reproducibility, and atomic failure
checks. Python tests exercise the same public contracts; both tutorials
demonstrate native training. Supported features are present in C++ and
the `pyariadne` binding. Full cross-platform CI remains the acceptance
gate; performance, parallel batch execution, persistence/transfer
learning, and validated training guarantees remain outside M4.
No serialization format is implemented: the agreed M5 storage plan
remains versioned JSON plus Safetensors and ONNX for portability.

### M5 — Persistence, freezing, and transfer learning

**Deliverables:** Backend-neutral versioned canonical JSON manifest plus
Safetensors parameter checkpoint, following the agreed open-format decision
above; ONNX is the separate portable model-exchange interface. Specify exact
binary64 preservation, schema compatibility and explicit conversion failures.
Additionally implement the originally planned
exact topology/activation metadata, typed parameter import/export,
compatibility checks, frozen parameter groups, replacement of output heads,
fine-tuning, and clear handling of optional backend-specific optimizer state
and reproducible model identity.

**Acceptance:** Round trips preserve serialized parameter values and
semantics as specified; intentional incompatible loads fail explicitly;
frozen weights do not change after optimizer steps; both language
tutorials demonstrate a small fine-tuning workflow. A certificate
attached to a snapshot must be invalidated by relevant changes.

### M6 — Validated inference and numerical enclosures

**Deliverables:** Implement a distinct Ariadne validated evaluator consuming
frozen canonical snapshots, with rigorous input-box output enclosures via
interval bound propagation using Ariadne's directed-rounding/validated numeric
facilities.
Extend to derivative enclosures and tighter affine/Taylor propagation
only after basic inclusions work. Make exact model parameters and their
conversion into validation arithmetic explicit.

**Acceptance:** Bounds are tested against analytic corner cases and
independent enclosure oracles, including cancellation and activation
extrema. Sampling is an additional falsification technique, not a proof.
No floating-point approximate result may be silently relabeled
validated. Document domain, parameter, rounding, and backend assumptions.

### M7 — Structural constraints and construction guarantees

**Deliverables:** Constraints such as nonnegative weights and bounded
parameters, monotone subnetworks with explicit activation conditions,
conservative Lipschitz certificates, and selected bounded-output
parameterizations. Begin with transformations that preserve feasibility
at every optimizer update; add more general constrained learning only
when justified by an actual use case.

**Acceptance:** Each construction has a mathematical statement of the
property, documented hypotheses and domain, direct invariant tests
across initialization/training/checkpoint transfer, and no reliance on
soft loss penalties as a proof. Generic LP/NLP dependencies are not
introduced without an explicit design and dependency-DAG review.

### M8 — Function interoperability and Reasoning readiness

**Deliverables:** A Function-compatible view of frozen networks and,
where appropriate, specialized numerical bounds and input derivatives.
Prove API boundaries and smoothness expectations rather than assuming
that every activation is differentiable. Provide a standalone
composition example that does not make Neural depend on Reasoning, SMT,
System, or Evolution.

**Acceptance:** Small smooth networks yield consistent native,
Function-based, and validated evaluations. Record exact conversion
semantics and performance costs. The existing Barr3 fixture is a
follow-on integration/performance benchmark for Reasoning or SMT;
an `UNKNOWN` solver result is not an accepted certificate.

### M9 — Performance and optional backend completion

**Deliverables:** Profile and improve CPU kernels and batching without
changing the canonical model; compare supported worker counts and
approximate/validated evaluation costs. Complete optional LibTorch/MLX C++
implementations behind the backend contract defined in M1, with capability
checks, conversion tests, and independent optional build targets. Perform
small, scoped compatibility spikes after M2–M4 instead of discovering
structural incompatibilities only at the end.

**Acceptance:** Benchmarks report workload, precision, hardware, baseline,
error tolerance, and numerical equivalence criteria. No mandatory new
machine-learning dependency; accelerated inference must not inherit
validation claims without separate error analysis or certification.

### Milestone implementation order

```text
M0 baseline CI
  -> M1 model
  -> M2 CPU forward
  -> M3 derivatives
  -> M4 training
  -> M5 persistence / transfer
  -> M6 validated enclosures
  -> M7 structural guarantees
  -> M8 Function interoperability / Reasoning readiness
  -> M9 optional optimization / backends
```

Some interfaces (especially numeric semantics, parameter freezing,
backend capabilities, external framework conversions, and the Function
adapter) may need small proof-of-concept tests in an earlier milestone to
detect architectural incompatibilities. Such experiments
must remain bounded and their results must be recorded here; they do
not change the committed dependency direction.

## Implementation documentation discipline

**Mandatory for every implementation increment:** Modify
`doc/neural_design.md` **in the same commit or pull request as the
implementation**, not in a later undocumented follow-up. Keep this
document as the design journal and living contract. Each increment
must add or update an entry in the implementation log below covering:

- **Scope and status:** milestone ID, implementation completed, and public
  API changes, including any explicit deviations from the original plan.
- **Design decisions:** mathematical model, data representation,
  numerical assumptions, precision, rounding, concurrency, and relevant
  trade-offs. Record rejected alternatives when they affect later work.
- **Tests and evidence:** exact new or changed tests and reproducible
  commands. Note CI success, failures, or checks still pending truthfully.
- **Limitations and next step:** known edge cases, unsupported features,
  remaining proof obligations, and what is required before proceeding.

Update the existing design sections whenever an implemented choice
supersedes a proposed one. Mark claims as **planned**, **implemented**,
**tested**, or **formally established** as appropriate; test success and
sample checks alone are not mathematical proof. C++ and Python source files
must carry the standard Ariadne Neural copyright/GPL header; CMake files
must **not** carry this header.

### Implementation log

| Date (UTC) | Milestone | Status | Changes, evidence, limitations |
| --- | --- | --- | --- |
| 2026-11-03 | M4 frozen-session capability contract | Fixed; CI pending | Debug Python CTest on Ubuntu/macOS/Windows exposed two expectations that frozen NeuralExecutionSession rejects Training. M4 had incorrectly made its per-session supports()/require() accept Training merely because backend-wide NativeCpu supports it. Restored the intended distinction: neural_backend_supports(NativeCpu, Training) remains true, while frozen sessions reject Training and only mutable NeuralTrainingSession performs updates. Added both sides of this contract to C++ and Python assertions; no training algorithm, numerical oracle, or coverage removed. |
| 2026-11-03 | M4 sequential CPU training | Implemented; CI pending | Added separate mutable training workspace, sample-mean MSE and stable BCE with logits, SGD with momentum/coupled decay, AdamW with decoupled decay, deterministic Xavier initialization and shuffled sample indices. C++/Python tests and tutorials cover independent step oracles, convergence, shape errors and frozen snapshot/failed-step invariants. No parallel training, persistence or validation guarantee; rerun platform CI. |
| 2026-11-03 | Model storage/exchange format decision | Design agreed; not implemented | Selected ONNX for external model portability and versioned JSON manifest + Safetensors for canonical Neural storage. Required binary64-preserving internal round trips, semantic checks for ONNX, clear rejection of unsupported operators/metadata and no mandatory ML runtime. Schema, identity digest, dependencies, parsers, and tests remain M5 implementation work; this change contains no runtime code. |
| 2026-11-03 | M3 Python CI test oracle | Corrected; CI pending | Python CTest discovery in M3 exposed a pre-existing malformed M1 topology assertion in `python/tests/test_network.py`: the second layer's canonical `biases_offset` is 12 for a 2→3→1 network (weights offset 9, three weights), but the expected tuple omitted that fourth field. Updated only the expected metadata tuple; verified it against the M1 row-major offset formula and reported CI actual output. Model, bindings and semantic coverage unchanged; rerun all CI, especially Python CTest. |
| 2026-11-03 | Windows CI sccache startup | CI mitigation; verification pending | On Windows x64/MSVC the Debug build failed before compiler execution because sccache timed out starting its server while compiling transitive Utility `command_line_interface.cpp`; this is a compiler-launcher failure, not evidence of a Utility source defect. Disabled sccache only in `.github/workflows/win.yml` for both main build and installed C++ tutorial, allowing direct MSVC compilation. No production code, test assertion, compiler warnings, matrix or dependency pin changed. Validate all four Windows matrix jobs on the new main commit; note possible slower uncached builds. |
| 2026-11-03 | M3 CPU autodiff | Implemented; CI pending | Native CPU reverse-mode input and canonical parameter gradients, full input Jacobian, batch sum/mean VJP accumulation and ReLU-at-zero subgradient=0; C++ analytic and finite-difference tests, Python tests, tutorial examples, README and Python CTest discovery correction. No training or validated claims; run all platform CI. |
| 2026-11-03 | Bootstrap | Implemented (CI under observation) | Project skeleton, pinned Configuration/Testing/Function/Threading, source and binding smoke tests, shared `tutorial/` directory, CI workflows. No neural model or training implementation. |
| 2026-11-03 | M0 | Planned / investigation open | Linux/GCC coverage job fails in completed earlier runs; macOS coverage passes. Establish root cause and fix before M1. |
| 2026-11-03 | Planning | Recorded | Milestones M0–M9 and mandatory per-implementation design-log update policy established; no new runtime functionality in this change. |
| 2026-11-03 | Multi-backend architecture | Design agreed; not implemented | Split canonical model/snapshot, execution backend and high-level operations; plan native CPU, Ariadne validated, optional LibTorch C++ and MLX C++ implementations. M1 must define backend capabilities and data conversion; no runtime API or code added yet. |
| 2026-11-03 | M1 core implementation | Implemented; CI pending | Added `NeuralNetwork`, `NeuralDenseLayerSpec`, `NeuralDenseLayerLayout`, `NeuralParameterSnapshot`, `NeuralExecutionSession`, backend/operation enums and truthful capability queries. Added C++ `test_network`/`test_backend`, Python `test_network.py`, matching tutorial updates and umbrella headers. No inference, gradients, training, or persistence yet. Validation: positive dimensions, nonempty layers, known activations, overflow-checked weight/bias offsets, finite binary64 parameters, matching array lengths, and rejection of unsupported backends. |

| 2026-11-03 | M2 compile integration fix | Implemented; full CI pending | Corrected the initial CPU source header resolution, and explicitly rejected unexpected activations with a `default` case so `-Wswitch-default -Werror` remains clean. Trigger: macOS arm64 Clang failed before tests at missing header; verify all platform CI at this commit. |
| 2026-11-03 | M2 CPU inference | Implemented; full CI pending | Added NativeCpu single/batch forward API, row-major FMA kernels, stable activation functions, input and intermediate checks; C++/Python tests and tutorials. No gradients or validated evaluation. Windows regression must be rerun and diagnosed. |
| 2026-11-03 | Testing performance regression | Dependency propagation; CI pending | Updated Testing, Function, and Threading gitlinks after fixing eager operand rendering for successful console assertions. Numeric oracle checks are unchanged; no neural architecture or runtime semantics altered. |

## Engineering and CI policy

Follow the modular Ariadne layout: flat `include/neural`, flat `src`, `tests`,
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
