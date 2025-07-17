import math
from matplotlib import pyplot as plt
import numpy as np
from kinetics import Kinetics

def draw_full_plot(t, dose_ng, user_dose: float, substance: dict, xticks_tspan: float):
    plt.plot(t, dose_ng)
    plt.xlabel("Time (hours)")
    
    step = math.floor(xticks_tspan/24)
    plt.xticks(np.arange(start=0, stop=xticks_tspan+1, step=step))

    plt.ylabel(f"{substance['active']} (ng/mL)")
    if 'prodrug' in substance:
        plt.title(f"Plasma {substance['active']} after {user_dose} mg of {substance['prodrug']}")
    else:
        plt.title(f"Plasma {substance['active']} after {user_dose} mg")
    plt.grid(True)
    plt.show()

def plot_last_dose(calc_func: callable, model_func: callable, y0, user_dose: float, substance: dict, xticks_tspan: float, kinetics: Kinetics):
    _, t, dose_ng = calc_func(model_func, kinetics, 0, xticks_tspan, user_dose, y0)
    plt.plot(t, dose_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel(f"{substance['active']} (ng/mL)")
    plt.xticks(np.arange(xticks_tspan+1))

    plt.ylabel(f"{substance['active']} (ng/mL)")
    if 'prodrug' in substance:
        plt.title(f"Plasma {substance['active']} after {user_dose} mg of {substance['prodrug']}")
    else:
        plt.title(f"Plasma {substance['active']} after {user_dose} mg")
    plt.grid(True)
    plt.show()

def test_plot(t, Cdex_ng, dose: float, tmax_target: float, cmax_target: float):
    plt.plot(t, Cdex_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("(ng/mL)")
    plt.axhline(cmax_target, color='gray', linestyle='--', label='Cmax target')
    plt.axvline(tmax_target, color='red', linestyle='--', label='Tmax target')
    plt.title(f"Plasma concentration after {dose} mg")
    plt.grid(True)
    plt.show()