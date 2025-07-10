from collections.abc import Sequence
import numpy as np
from matplotlib import pyplot as plt
from optimizeModel import solve_odes_michaelis, optimize_michaelis_menten_kinetics
from kinetics import Kinetics

def ldx_model(t, y, ka, ke, ke_ldx, f, Vmax, Km):
    ldx_gi, ldx_p, dex_p = y

    conv = (Vmax * ldx_p) / (Km + ldx_p)
    
    ldx_gi_dt = -ka * ldx_gi
    ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
    dex_p_dt = f * conv - (ke * dex_p)
        
    return (ldx_gi_dt, ldx_p_dt, dex_p_dt)

def get_result_ldx(kinetics: Kinetics, model_func, t_span: tuple[float, float], dose_ug: float,
                   optimize_start: Sequence[float, float]):
    result = optimize_michaelis_menten_kinetics(
        optimize_start,
        model_func,
        t_span,
        kinetics.Vd,
        kinetics.ke,
        kinetics.ka,
        kinetics.efficiency,
        dose_ug,
        kinetics.Tmax,
        kinetics.Cmax,
        kinetics.auc,
        kinetics.ke_pro
    )
    return result

def calculate_curve(model_func, kinetics: Kinetics, t_start: float, t_end: float, dose_mg: float,
                    y0: Sequence[float] = None) -> tuple:
    if y0 is None:
        y0 = [dose_mg*1000, 0, 0]
        
    t_span = (t_start, t_end)

    solution = solve_odes_michaelis(
        model_func, t_span, y0, 
        kinetics.ka, 
        kinetics.ke, 
        kinetics.ke_pro, 
        kinetics.efficiency, 
        kinetics.Vmax, 
        kinetics.Km)
    
    t = solution.t
    y = solution.y
    
    Cdex_ng = (y[2] / kinetics.Vd) * 1000

    return (y, t, Cdex_ng)

def draw_full_plot(t, dose_ng, user_dose: float, plot_tspan: float = None, x_left_lim: float = None,
                     x_right_lim: float = None):
    plt.plot(t, dose_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("d-Amphetamine (ng/mL)")
    plt.minorticks_on()

    if x_left_lim != None:
        plt.xlim(left=x_left_lim)
    elif plot_tspan != None:
        plt.xlim(left=t[-1]-plot_tspan)

    if x_right_lim != None:
        plt.xlim(right=x_right_lim)
    else:
        plt.xlim(right=t[-1]+t[-1]*0.01)
    
    plt.title(f"Plasma d-Amphetamine after {user_dose} mg LDX")
    plt.grid(True)
    plt.show()

def plot_last_dose(y0, user_dose: float, plot_tspan: float, kinetics: Kinetics):
    _, t, dose_ng = calculate_curve(ldx_model, kinetics, 0, plot_tspan, user_dose, y0)
    plt.plot(t, dose_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("d-Amphetamine (ng/mL)")
    plt.minorticks_on()
    plt.title(f"Plasma d-Amphetamine after {user_dose} mg LDX")
    plt.grid(True)
    plt.show()

def test_plot(t, Cdex_ng, dose: float, tmax_target: float, cmax_target: float):
    plt.plot(t, Cdex_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("d-Amphetamine (ng/mL)")
    plt.axhline(cmax_target, color='gray', linestyle='--', label='Cmax target')
    plt.axvline(tmax_target, color='red', linestyle='--', label='Tmax target')
    plt.title(f"Plasma d-Amphetamine after {dose} mg LDX")
    plt.grid(True)
    plt.show()

def get_user_input() -> tuple:
    multiple_dose = float(input("multiple doses(yes/no [1/0]): "))
    if multiple_dose > 0:        
        t_doses = list(map(float, input("times between the doses in hours (separated by commas): ")
                              .split(',')))       
    else:
        t_doses = None
    dose_mg = float(input("dose of LDX in mg: "))
    t_end = float(input("end of curve timespan in hours: "))

    return (t_end, dose_mg, t_doses)

def simulate(model_func, kinetics: Kinetics, t_end: float, dose_mg: float, t_doses: None|list):
    if t_doses is None:
        _, t, ng = calculate_curve(model_func, kinetics, 0, t_end, dose_mg)
    else:
        t_doses.append(t_end)
        y, t, ng = calculate_curve(model_func, kinetics, 0, t_doses[0], dose_mg)
        y_all, t_all, ng_all = y, t, ng
        t_next = t_doses[0]
        for i in range(len(t_doses)-1):
            y, t, ng = calculate_curve(model_func, kinetics, t_next, t_next+t_doses[i+1], dose_mg,
                                    y0=[dose_mg*1000, y[1][-1], y[2][-1]])
            
            t_next = t_next + t_doses[i+1]
            y_all = np.concatenate([y_all, y])
            t_all = np.concatenate([t_all, t], axis=None)
            ng_all = np.concatenate([ng_all, ng], axis=None)

    return (y_all, t_all, ng_all)