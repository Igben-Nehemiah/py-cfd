import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Data
L = 2                      # Length in cm
k = 10                     # Thermal conductivity in W/m.K
rho_c = 10e7              # Product of density and specific heat capacity in J/m3.K

init_temp = 200            # Initial temperature in degree celcius
time_interval = 4          # Time interval in seconds


n_nodes = 5                # Number of control volume cells
dx = (L/100)*(1/5)         # Cell width in m
dt = 1                     # Time delta in s

n_time_steps = int(time_interval/dt )

x_A = 4
x_B = 1

T = np.full((n_nodes + 2, ), init_temp) # With ghose cells

for t in range(n_time_steps):

    aW = aE = k/dx
    aP = rho_c * dx/dt
    
    main_diagonal = np.full((n_nodes + 2, ), aP) * T
    sup_diagonal = np.full((n_nodes + 1, ), -aE) * T[1:]
    sub_diagonal = np.full((n_nodes + 1, ), -aW) * T[:-1]

    b = np.zeros_like(main_diagonal)

    main_diagonal[0] = 10e30
    b[0] = 10e30*x_A
    main_diagonal[-1] = 10e30
    b[-1] = 10e30*x_B

    A = diags([main_diagonal, sup_diagonal, sub_diagonal], [0, 1, -1], format='csr')

    x2 = spsolve(A, b)
    print(x2)
