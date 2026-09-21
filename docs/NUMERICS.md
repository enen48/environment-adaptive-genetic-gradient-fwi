# Numerical contract

The reference backend solves the constant-density acoustic equation
`u_tt = v^2 * Laplacian(u) + f` on a Cartesian 2D grid using second-order
central differences. Source amplitudes are discrete update increments rather
than calibrated pressure/force units. Observed and predicted traces must use
the same convention.

## Stability and boundaries

For equal grid spacing, the second-order 2D stencil requires
`max(v) * dt / dx <= 1/sqrt(2)`; the implementation checks a slightly smaller
safety limit and fails instead of hiding an unstable simulation. Velocities
must be finite, positive and expressed in m/s, spacing in m and time in s.
Receivers and sources use integer `(depth, horizontal)` grid coordinates.

The minimal backend adds a padded sponge outside the physical velocity grid.
The `pml_width` configuration name denotes its width for future backend
compatibility; this sponge is **not a true PML**. The outer computational edge
has a zero-value boundary and can still reflect. Reproduction of field data
or OpenFWI's published waveforms requires a matched, more accurate solver.

[Deepwave's scalar formulation](https://www.ausargeo.com/deepwave/scalar)
documents a suitable replacement backend with a true PML. This project does
not claim the reference backend has that behavior.

## Differentiation and frequency filtering

The full latent refinement chain remains in PyTorch:
`z -> D(z) -> smoothed velocity -> acoustic propagation -> receiver traces
-> FFT filtering -> residual -> loss`. Frozen AE weights do not imply
`no_grad()` around the decoder during local refinement.

All bands use a shared forward trace. The reference implementation provides
FFT masks, cosine transition (smooth/Tukey-like) masks and Butterworth-shaped
frequency responses. The Butterworth option is a zero-phase spectral response,
not a causal IIR implementation. Zero padding reduces wrap-around but does not
add physical frequency resolution to a short trace. CPU smoke windows are
therefore insufficient to establish realistic low-frequency performance.

PyTorch FFT operations retain autograd support; see the
[official torch.fft documentation](https://docs.pytorch.org/docs/stable/fft.html).

## Cost and changing objectives

Track model forward evaluations, shot-equivalent propagations, and Python
solver invocations separately. A batched call can represent several models
and shots. Count generation, reference-monitoring, initialization and local
refinement calls; report observation generation separately from optimization.

Current-environment fitness is used for selection **within a generation**.
It cannot be used as a comparable all-time minimum while frequency weights,
velocity smoothing and regularization change. A fixed final-environment
reference objective selects the final incumbent and controls stagnation.
This reference monitoring costs additional PDE evaluations, which must be
included in the count. It is an optimization monitor, not a held-out
validation dataset.

Z-bank entries retain their source environment and a separate fixed-reference
fitness. Selection normalizes that reference fitness and penalizes distance
from the current environment; every injected gene is re-evaluated before
selection. No historical raw fitness is reused as current fitness. Memory
initialization uses AE training data only and marks observed-data quality
unknown. Unknown-quality seeds may initialize the population but are excluded
from online memory injection until evaluated.

By default every feasible candidate is monitored at the fixed reference, and
all extra forwards count toward cost. Setting `reference_top_k` explicitly
enables a cheaper approximate monitor; it can miss a candidate that is poor
under the current environment yet strong under the reference environment.
The default stopping warmup (`min_progress`) lets the environment reach finer
frequencies before counting reference stagnation.

## Reproducibility

Inversion checkpoints must include frozen AE weights, next-generation
population, Z-bank, QC calibration, current/reference best, complete RNG
states, stopping state, pending injection assessment, counters, observations,
acquisition geometry and the effective configuration. Resume uses the same
generation horizon; changing it would change the continuous schedule. Runtime
will differ across resumed runs even when numerical tensors match.

The baseline and ablations share each seed's observations and an initial model.
Their default iteration budgets differ in actual work; compare PDE counts and
misfit, not only wall time or generation number. Repeated seeds and a separate
held-out benchmark are required before making claims of algorithmic benefit.
