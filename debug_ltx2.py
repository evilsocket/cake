"""Debug: just compare scheduler sigmas between Rust and Python."""
import math

# LTX-2 config
base_shift = 0.95
max_shift = 2.05
num_steps = 30
num_tokens = 2112  # 6*16*22
power = 1.0
stretch_terminal = 0.1

# Compute mu (dynamic shift)
base_seq = 1024.0
max_seq = 4096.0
m = (max_shift - base_shift) / (max_seq - base_seq)
b = base_shift - m * base_seq
mu = num_tokens * m + b
print(f"mu = {mu:.6f}")

def flux_time_shift(mu, sigma, t):
    emu = math.exp(mu)
    if t <= 0.0 or t >= 1.0:
        return t
    base = (1.0/t - 1.0) ** sigma
    return emu / (emu + base)

# Generate N sigmas (no zero), apply shift
sigmas = []
for i in range(num_steps):
    s = 1.0 - i / num_steps
    s = flux_time_shift(mu, power, s)
    sigmas.append(s)

print(f"\nBefore stretch ({len(sigmas)} sigmas):")
print(f"  First 3: {sigmas[:3]}")
print(f"  Last 3: {sigmas[-3:]}")

# Stretch to terminal
last = sigmas[-1]
one_minus_last = 1.0 - last
denom = 1.0 - stretch_terminal
scale = one_minus_last / denom
for i in range(len(sigmas)):
    one_minus = 1.0 - sigmas[i]
    sigmas[i] = 1.0 - (one_minus / scale)

sigmas.append(0.0)

print(f"\nAfter stretch + append zero ({len(sigmas)} sigmas):")
for i, s in enumerate(sigmas):
    print(f"  sigma[{i:2d}] = {s:.6f}")

# Also print the timestep (1 - sigma) * 1000 for comparison
print(f"\nTimestep = (1 - sigma) * 1000:")
for i in range(len(sigmas)-1):
    print(f"  step {i:2d}: sigma={sigmas[i]:.6f}, timestep={(1-sigmas[i])*1000:.2f}")

# Check: are all sigmas monotonically decreasing?
for i in range(1, len(sigmas)):
    if sigmas[i] > sigmas[i-1]:
        print(f"  WARNING: sigma[{i}]={sigmas[i]} > sigma[{i-1}]={sigmas[i-1]}")

# Check: are all sigmas non-negative?
for i, s in enumerate(sigmas):
    if s < 0:
        print(f"  WARNING: sigma[{i}]={s} is negative!")
