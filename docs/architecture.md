# Architecture

`spikegen` is a small set of pure functions over the standard library `random` and `math`.
Each generator returns a sorted `list[float]` of spike times in `[0, duration)`.

## Regular

`regular` places spikes at `0, 1/rate, 2/rate, ...`. The index is multiplied by the step
rather than accumulated, so floating-point error does not drift across a long train.

## Homogeneous Poisson

`homogeneous_poisson` draws exponential inter-spike intervals: the next interval is
`-ln(1 - U) / rate` for a uniform U. Using `1 - U` keeps the argument of the log away from
zero. Spikes accumulate until the time leaves `[0, duration)`.

## Inhomogeneous Poisson

`inhomogeneous_poisson` uses thinning: propose spikes from a homogeneous process at
`max_rate`, then keep each proposed spike at time t with probability `rate_fn(t) / max_rate`.
`rate_fn` must return a value in `[0, max_rate]`; a value outside that range raises, since
it would make the thinning probability invalid.

## Gamma renewal

`gamma_renewal` draws inter-spike intervals from a gamma distribution with the given shape
and a scale of `1 / (rate * shape)`, so the mean interval is `1 / rate`. Shape 1 reduces to
the exponential intervals of a Poisson process; larger shapes give more regular spiking.

## Refractory filter

`with_refractory` sorts the input and keeps a spike only when it is at least `refractory`
after the previously kept spike, which models an absolute refractory period.

## Population

`population` builds a list of spike trains by calling a user-supplied `make(seed)` once per
unit. It draws all child seeds up front from a single `random.Random(seed)`, then calls
`make` with each. Drawing the seeds before any train is generated keeps each unit independent
and makes the whole population a function of `(seed, units)` only, regardless of how much
randomness `make` consumes. A deterministic generator that ignores its seed yields identical
units, which is the expected behavior.

## Bernoulli

`bernoulli` tiles `[0, duration)` into `int(duration / dt)` bins of width dt. Each bin's
start time is emitted as a spike with probability `p = rate * dt`, sampled from a Bernoulli
draw using `rng.random() < p`. The constraint `p <= 1` ensures a valid probability; the
function raises `ValueError` with an explicit message when `rate * dt > 1`. Using integer
bin indices (`i * dt`) avoids accumulated floating-point drift across a long train.

## Jitter

`jitter` adds an independent Gaussian deviate (mean 0, standard deviation sigma) to each
input spike time using `rng.gauss(0.0, sigma)`, then sorts the result. The output is not
constrained to `[0, duration)` because the input is an arbitrary spike sequence, not a
bounded generative process; callers that need boundary enforcement can apply `with_refractory`
or filter manually. `sigma = 0` skips the RNG and returns `sorted(times)` directly. The
primary use case is generating surrogate spike trains for null-hypothesis testing by
destroying precise timing relationships while preserving the overall spike count.

## Reproducibility

Every stochastic generator takes an integer `seed` and uses `random.Random(seed)`, so the
same arguments always produce the same train. The tests rely on this for exact assertions
and confirm that different seeds give different trains.

## Why pure Python

The processes are short and standard, and the standard library provides uniform and gamma
variates. Implementing them with no dependencies keeps installation trivial and the output
(plain lists of floats) immediately usable, including by spikedist.
