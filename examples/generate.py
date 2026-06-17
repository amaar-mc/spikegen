"""Generate a few spike trains and print their spike counts.

Run with: python examples/generate.py
"""

from spikegen import gamma_renewal, homogeneous_poisson, regular, with_refractory

duration = 1.0

poisson = homogeneous_poisson(rate=50.0, duration=duration, seed=0)
gamma = gamma_renewal(rate=50.0, shape=4.0, duration=duration, seed=0)
clock = regular(rate=50.0, duration=duration)
refractory = with_refractory(poisson, refractory=0.005)

print(f"duration: {duration} s, target rate: 50 Hz")
print(f"  regular            {len(clock)} spikes")
print(f"  poisson            {len(poisson)} spikes")
print(f"  gamma (shape 4)    {len(gamma)} spikes")
print(f"  poisson + 5 ms ref {len(refractory)} spikes")
