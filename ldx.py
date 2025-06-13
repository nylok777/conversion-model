import numpy as np
from optimizeModel import solve_odes, get_michaelis_menten_params
from matplotlib import pyplot as plt

class LdxKinetics():
    def __init__(self):
        self.ke = np.log(2) / 10
        self.ka = 0.6691
        self.ke_ldx = 0.9
        self.f = 0.297
        self.Vd = 195_000
        self.Tmax = 3.78
        self.Cmax = 69.3
        self.auc = 1020
        self.optimize_start = [35000, 30000]
        self.Vmax = 0
        self.Km = 0
    
    def set_michaelis_params(self):
        result = get_result
        self.Vmax = result.x[0]
        self.Km = result.x[1]

def ldx_model(t, y, ka, ke, ke_ldx, f, Vmax, Km):
    ldx_gi, ldx_p, dex_p = y

    conv = (Vmax * ldx_p) / (Km + ldx_p)
    
    ldx_gi_dt = -ka * ldx_gi
    ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
    dex_p_dt = f* conv - (ke * dex_p)
        
    return [ldx_gi_dt, ldx_p_dt, dex_p_dt]

def get_result(ldx_kinetics: LdxKinetics):

    result = get_michaelis_menten_params(
        ldx_kinetics.optimize_start,
        ldx_model,
        (0, 72),
        (20_000, 70_000),
        (10_000, 60_000),
        ldx_kinetics.Vd,
        ldx_kinetics.ke,
        ldx_kinetics.f,
        70*1000,
        ldx_kinetics.Tmax,
        ldx_kinetics.Cmax,
        ldx_kinetics.auc,
        ldx_kinetics.ke_ldx
    )
    return result

def calculate_curve(ldx_kinetics: LdxKinetics, t_end: float, dose_mg: float, t_continue: float | None = None, y0 = None):
    if y0 is None:
        y0 = [dose_mg*1000, 0, 0]
        
    if t_continue is None:
        t_span = (0, t_end)
    else:
        t_span = (t_continue, t_end)
    solution = solve_odes(
        ldx_model, t_span, y0, 
        ldx_kinetics.ka, 
        ldx_kinetics.ke, 
        ldx_kinetics.ke_ldx, 
        ldx_kinetics.f, 
        ldx_kinetics.Vmax, 
        ldx_kinetics.Km)
    
    t = solution.t
    y = solution.y
    
    Cdex_ng = (y[2] / ldx_kinetics.Vd) * 1000

    return (y, t, Cdex_ng)

def draw_plasma_plot(t, dose_ng, user_dose):
    plt.plot(t, dose_ng)
    #plt.axhline(cmax_target, color='gray', linestyle='--', label='Cmax target')
    #plt.axvline(tmax_target, color='red', linestyle='--', label='Tmax target')
    plt.xlabel("Time (hours)")
    plt.ylabel("d-Amphetamine (ng/mL)")
    plt.title(f"Plasma d-Amphetamine after {user_dose} mg LDX")
    plt.grid(True)
    plt.show()

def get_user_input():
    multiple_dose = float(input("multiple doses(yes/no [1/0]): "))
    if multiple_dose > 0:        
        t_continue = float(input("time between doses in hours: "))        
    else:
        t_continue = None
    dose_mg = float(input("dose of LDX in mg: "))
    t_end = int(input("end of curve timespan in hours: "))

    return (t_end, dose_mg, t_continue)

def show_plot_to_user(ldx_kinetics: LdxKinetics):
    inputs = get_user_input()
    _, t, ng = calculate_curve(ldx_kinetics, *inputs)
    draw_plasma_plot(t, ng, inputs[1])