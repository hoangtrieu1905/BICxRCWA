from rcwa_v2 import calculate_winding_number
import numpy as np

# Synthetic vortex of charge +1
grid_size = 15
phase_test = np.zeros((grid_size, grid_size))

for i in range(grid_size):
    for j in range(grid_size):
        x = j - grid_size // 2
        y = i - grid_size // 2
        if x != 0 or y != 0:
            phase_test[i, j] = np.arctan2(y, x)   # +1 vortex

# Charge -1
phase_test2 = -phase_test

q_minus = calculate_winding_number(phase_test2)
print(f"Phase 1: Synthetic -1 vortex → q = {q_minus}")   # should be -1

q_plus = calculate_winding_number(phase_test)
print(f"Phase 2: Synthetic +1 vortex → q = {q_plus}")   # should be +1

# No vortex
phase_test3 = np.zeros((grid_size, grid_size))
q_zero = calculate_winding_number(phase_test3)
print(f"Phase 3: No vortex → q = {q_zero}")              # should be 0