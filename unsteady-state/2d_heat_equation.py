"""
A copper plate is initially at a uniform temperature of 200°C. At a certain time
t = 0 the temperature of the east, north and south sides of the plate are suddenly reduced to 0°C. 
The west surface is kept at 200°C. Use the explicit finite volume methods to calculate the transient 
temperature distribution of the slab at time (i) t = 40 s, (ii) t = 80 s and (iii) t = 120 s. 
The data are: plate width W = 500 cm, Plate height H = 500 cm, thermal conductivity k = 10 W/m.K and rho = 8850 kg/m3 and c = 389 J/kg/k. 
"""
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Data
init_temp = 0
W = 0.5  # plate's width
H = 0.5  # plate's height
k = 398  # plate's thermal conductivity
rho = 8850  # plate's density in kg/m3
c = 389  # plate's specific heat capacity in J/kg/k


TIME_INTERVAL = 40  # time interval in seconds
dt = 0.05  # time step in seconds
n_time_steps = floor(int(TIME_INTERVAL / dt)) + 1

nx_nodes = 100  # number of x- nodes
ny_nodes = 100  # number of y- nodes

dx = W/nx_nodes  # node's control volume width
dy = H/ny_nodes  # node's control volume height

# Initial condition
time_temp_dist = np.empty((n_time_steps, nx_nodes, ny_nodes))
time_temp_dist[0] = np.ones((nx_nodes, ny_nodes)) * init_temp


# Boundary conditions
T_W = 200  # plate's western surface temperature
T_E = 200  # plate's eastern surface temperature
T_N = 0  # plate's northern surface temperature
T_S = 0  # plate's southern surface temperature


# Check convergence criterion
dt_critical = rho*c*dx**2*dy**2/(2*k*(dx**2+dy**2))

if (dt >= dt_critical):
    raise ArithmeticError(
        f'The time step is greater than or equal to the critical time step {dt_critical}')


a_E = a_W = k*dy/dx
a_N = a_S = k*dx/dy
a_Po = rho*c*dx*dy/dt
a_P = a_Po


def solve():
    t_step = 0
    while (t_step < n_time_steps - 1):
        t_step += 1  # Change time to current time

        # Previous time temperature nodes
        T_prev = time_temp_dist[t_step - 1]

        # Current time temperature nodes
        T = time_temp_dist[t_step]

        # --------------------------------Internal nodes--------------------------------------------

        T[1:-1, 1:-1] = (a_E*T_prev[1:-1, 2:] + a_W*T_prev[1:-1, 0:-2] + a_N*T_prev[0:-2, 1:-1] +
                         a_S*T_prev[2:, 1:-1] + (a_Po - (a_W + a_E + a_N + a_S))*T_prev[1:-1, 1:-1])/a_P

        # ---------------------------------Boundary nodes---------------------------------------------
        # West
        S_u = 2*k*dy*(T_W - T_prev[1:-1, 0])/dx
        T[1:-1, 0] = (a_E*T_prev[1:-1, 1] + a_N*T_prev[0:-2, 0] +
                      a_S*T_prev[2:, 0] + (a_Po - (a_E + a_N + a_S))*T_prev[1:-1, 0] + S_u)/a_P

        # East
        S_u = 2*k*dy*(T_E - T_prev[1:-1, -1])/dx
        T[1:-1, -1] = (a_W*T_prev[1:-1, -2] + a_N*T_prev[0:-2, -1] +
                       a_S*T_prev[2:, -1] + (a_Po - (a_W + a_N + a_S))*T_prev[1:-1, -1] + S_u)/a_P

        # North
        S_u = 2*k*dx*(T_N - T_prev[0, 1:-1])/dy
        T[0, 1:-1] = (a_E*T_prev[0, 2:] + a_W*T_prev[0, 0:-2] +
                      a_S*T_prev[0, 1:-1] + (a_Po - (a_E + a_W + a_S))*T_prev[0, 1:-1] + S_u)/a_P

        # South
        S_u = 2*k*dx*(T_S - T_prev[-1, 1:-1])/dy
        T[-1, 1:-1] = (a_E*T_prev[-1, 2:] + a_W*T_prev[-1, 0:-2] +
                       a_N*T_prev[-1, 1:-1] + (a_Po - (a_E + a_W + a_N))*T_prev[-1, 1:-1] + S_u)/a_P

        # -------------------------------------Corners------------------------------------------------
        # N-W
        S_u = 2*k*dy*(T_W - T_prev[0, 0])/dx + \
            2*k*dx*(T_N - T_prev[0,  0])/dy
        T[0, 0] = (a_E*T_prev[0, 1] + a_S*T_prev[1, 0] + (a_Po - (a_E + a_S)) *
                   T_prev[0, 0] + S_u)/a_P

        # S-W
        S_u = 2*k*dy*(T_W - T_prev[-1, 0])/dx + \
            2*k*dx*(T_S - T_prev[-1,  0])/dy
        T[-1, 0] = (a_E*T_prev[-1, 1] + a_N*T_prev[-2, 0] + (a_Po - (a_E + a_N)) *
                    T_prev[-1, 0] + S_u)/a_P

        # N-E
        S_u = 2*k*dy*(T_E - T_prev[0, -1])/dx + \
            2*k*dx*(T_N - T_prev[0,  -1])/dy
        T[0, -1] = (a_W*T_prev[0, -2] + a_S*T_prev[1, -1] + (a_Po - (a_W + a_S)) *
                    T_prev[0, -1] + S_u)/a_P

        # S-E
        S_u = 2*k*dy*(T_E - T_prev[-1, -1])/dx + \
            2*k*dx*(T_S - T_prev[-1,  -1])/dy
        T[-1, -1] = (a_W*T_prev[-1, -2] + a_N*T_prev[-2, -1] + (a_Po - (a_W + a_N)) *
                     T_prev[-1, -1] + S_u)/a_P


def plotheatmap(temp_dist, n):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {n*dt:.2f} s")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.xticks([])
    plt.yticks([])

    plt.pcolormesh(temp_dist, cmap=plt.cm.jet, vmin=0, vmax=200)
    plt.colorbar()

    return plt


def animate(time):
    plotheatmap(time_temp_dist[time], time)


solve()

anim = animation.FuncAnimation(
    plt.figure(), animate, interval=1, frames=n_time_steps, repeat=False)


plt.show()
# if __name__ == '__main__':
# pass
