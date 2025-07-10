from collections.abc import Sequence
import numpy as np
from matplotlib import pyplot as plt
from kinetics import Kinetics
from optimizeModel import solve_odes_fo

def first_order_model(t, y, ka, ke):
    Agi, Ap = y

    d_Agi_dt = -ka*Agi
    d_Ap_dt = ka*Agi * np.e**(-ke*t) - (ke * Ap)

    return (d_Agi_dt, d_Ap_dt)

def calculate_curve(model_func, kinetics: Kinetics, t_start: float, t_end: float, dose_mg: float, y0: Sequence[float] = None):
    if y0 is None:
        y0 = [dose_mg*1000, 0]
        
    t_span = (t_start, t_end)

    solution = solve_odes_fo(
        model=model_func,
        t_span=t_span,
        y0=y0,
        ka=kinetics.ka,
        ke=kinetics.ke
    )

    y, t = solution.y, solution.t
    conc_ng = (y[-1] / kinetics.Vd) * 1000

    return (y, t, conc_ng)

def test_plot(t, conc_ng, dose: float, tmax_target: float, cmax_target: float):
    plt.plot(t, conc_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("substance (ng/mL)")
    plt.axhline(cmax_target, color='gray', linestyle='--', label='Cmax target')
    plt.axvline(tmax_target, color='red', linestyle='--', label='Tmax target')
    plt.title(f"Plasma concentration after {dose} mg of substance")
    plt.grid(True)
    plt.show()