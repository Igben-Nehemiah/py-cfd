'''
This contains a simple implementation of the upwind differencing method
'''

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

L = 1.5
H = 1.5
nx_nodes = ny_nodes = 45

diffusivity = 0.03      # kg/m.s
rho = 1                 # kg/m3
u = 2                   # x-component of velocity in m/s
v = 2                   # y-component of velocity in m/s

# Initial condition
init_temp = 100
T = np.full((nx_nodes, ny_nodes), init_temp)

TIME_INTERVAL = 40 
dt = 1

n_time_steps = int(TIME_INTERVAL/dt)

dx = L / nx_nodes
dy = H / ny_nodes

aP_0 = rho*dx*dy/dt   # TODO: Look into this later

Aw = Ae = dy
As = An = dx

Fw = rho * u * Aw
Fe = rho * u * Ae

Fn = rho * v * An
Fs = rho * v * As

Dw = diffusivity * Aw / dx 
De = diffusivity * Ae / dx
Dn = diffusivity * An / dy 
Ds = diffusivity * As / dy

aW = max(Fw, (Dw + 0.5 * Fw), 0)
aE = max(-Fe, (De - 0.5 * Fe), 0)
aS = max(Fs, (Ds + 0.5 * Fs), 0)
aN = max(-Fn, (Dn - 0.5 * Fn), 0)

dF = Fe - Fw + Fn - Fs           # A statement of continuity

n_cells = nx_nodes * ny_nodes

for _ in range(n_time_steps):
    S_P = 0

    aP = aW + aE + aS + aN + aP_0 + dF - S_P

    # Internal nodes 
    main_diagonal = np.full((n_cells,), aP)
    sub_diagonal_1 = np.full((n_cells - 1,), -aW)
    sub_diagonal_2 = np.full((n_cells))

    pass