# Temporary diagnostic — paste into rcwa_v2.py, run once, then delete
import numpy as np
from rcwa_v2 import get_reflection_with_T0R


# Monkey-patch the dielectric to be vacuum
# Easiest: edit get_reflection's ep[1] = 12.0 line temporarily to ep[1] = 1 + 1j*1e-9
# Then call any function that returns T0R, or modify get_reflection to also return T0R:

# Run with vacuum slab (after patching ep[1] = 1 + 1j*1e-9):
T0R = get_reflection_with_T0R(np.radians(9.0), 600.0, fourierMode=10)

# Print only the entries with magnitude > 0.01
M = 10
N = 4*M + 2   # = 42
for i, v in enumerate(T0R[:, 0]):
    if abs(v) > 0.01:
        block = "block_0 (idx 0-41)" if i < N else "block_1 (idx 42-83)"
        local = i if i < N else i - N
        if local < 2*M+1:
            sub_block, n = "s-pol", local - M
        else:
            sub_block, n = "p-pol", local - (2*M+1) - M
        print(f"  T0R[{i:3d}] = {v.real:+.6f} {v.imag:+.6f}j  |v|={abs(v):.4f}  → {block}, {sub_block}, n={n}")