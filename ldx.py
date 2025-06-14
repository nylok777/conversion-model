import numpy as np
from optimizeModel import solve_odes, get_michaelis_menten_params
from matplotlib import pyplot as plt
from kinetics import Kinetics

def ldx_model(t, y, ka, ke, ke_ldx, f, Vmax, Km):
    ldx_gi, ldx_p, dex_p = y

    conv = (Vmax * ldx_p) / (Km + ldx_p)
    
    ldx_gi_dt = -ka * ldx_gi
    ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
    dex_p_dt = f * conv - (ke * dex_p)
        
    return (ldx_gi_dt, ldx_p_dt, dex_p_dt)

def get_result_ldx(kinetics: Kinetics, model_func):

    result = get_michaelis_menten_params(
        kinetics.optimize_start,
        model_func,
        (0, 72),
        (20_000, 70_000),
        (10_000, 60_000),
        kinetics.Vd,
        kinetics.ke,
        kinetics.ka,
        kinetics.f,
        70*1000,
        kinetics.Tmax,
        kinetics.Cmax,
        kinetics.auc,
        kinetics.ke_pro
    )
    return result

def calculate_curve(
        model_func, 
        kinetics: Kinetics,
        t_start: float,
        t_end: float, 
        dose_mg: float,         
        y0 = None
        ):
    
    if y0 is None:
        y0 = [dose_mg*1000, 0, 0]
        
    t_span = (t_start, t_end)

    solution = solve_odes(
        model_func, t_span, y0, 
        kinetics.ka, 
        kinetics.ke, 
        kinetics.ke_pro, 
        kinetics.f, 
        kinetics.Vmax, 
        kinetics.Km)
    
    t = solution.t
    y = solution.y
    
    Cdex_ng = (y[2] / kinetics.Vd) * 1000

    return (y, t, Cdex_ng)

def draw_plasma_plot(t, dose_ng, user_dose):
    plt.plot(t, dose_ng)
    plt.xlabel("Time (hours)")
    plt.ylabel("d-Amphetamine (ng/mL)")
    plt.title(f"Plasma d-Amphetamine after {user_dose} mg LDX")
    plt.grid(True)
    plt.show()

def get_user_input():
    multiple_dose = float(input("multiple doses(yes/no [1/0]): "))
    if multiple_dose > 0:        
        t_continue = list(map(float, input("times between the doses in hours (separated by commas): ")
                              .split(',')))       
    else:
        t_continue = None
    dose_mg = float(input("dose of LDX in mg: "))
    t_end = float(input("end of curve timespan in hours: "))

    return (t_end, dose_mg, t_continue)

def show_plot_to_user(model_func, kinetics: Kinetics):
    t_end, dose_mg, t_doses = get_user_input()

    if t_doses is None:
        _, t, ng = calculate_curve(model_func, kinetics, 0, t_end, dose_mg)
    else:
        t_doses.append(t_end)
        y, t, ng = calculate_curve(model_func, kinetics, 0, t_doses[0], dose_mg)
        for i in range(len(t_doses)-1):
            y, t, ng = calculate_curve(model_func, kinetics, t_doses[i], t_doses[i]+t_doses[i+1], dose_mg,
                                    y0=[dose_mg*1000, y[1][-1], y[2][-1]])

    draw_plasma_plot(t, ng, dose_mg)