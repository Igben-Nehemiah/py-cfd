import numpy as np

def divergence_field(x: float, y: float):
    return 2*x, 2*y


def lotka_volterra(x:float, y:float, alpha:float, beta:float, delta:float, gamma:float):
    """
    The Lotka-Volterra model describes the dynamics of a predator-prey system and can be 
    represented as a system of ordinary differential equations (ODEs). 
    To represent the solution of this model as a vector function, you can use a vector notation.

    
    `dx/dt = α * x - β * x * y `

    `dy/dt = δ * x * y - γ * y`

    Here:
    - `dx/dt` and `dy/dt` represent the rates of change of the prey and predator populations over time, respectively.

    - α (prey growth rate): This constant represents the intrinsic growth rate of the prey 
      population in the absence of predators. It can vary depending on the species of prey and 
      environmental conditions. For example, for rabbits, it might be in the range of 0.1 to 0.5 per month.

    - β (predation rate coefficient): β represents the rate at which predators consume the prey. 
      Its value depends on the predator's efficiency in hunting the prey and the availability of 
      other food sources. A common range might be 0.01 to 0.05 per (prey * predator).

    - δ (conversion efficiency of prey to predator): δ represents how efficiently the predators 
      convert consumed prey into increased population. It typically ranges from 0.01 to 0.1.

    - γ (predator death rate): γ represents the predator death rate. It can be influenced by factors 
      like mortality due to old age, diseases, or other predators. A typical range might be 0.1 to 0.4 per month.


    To represent this system as a vector function F(x, y), you can use the following notation:

    `F(x, y) = [dx/dt, dy/dt]`

    `F(x, y) = [α * x - β * x * y, δ * x * y - γ * y]`

    """
    return alpha*x-beta*x*y, delta*x*y-gamma*y