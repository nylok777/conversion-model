from collections.abc import Sequence
from collections import deque
import numpy as np
from optimizeModel import solve_odes_michaelis, optimize_michaelis_menten_kinetics
from kinetics import KineticsFromProDrug

def ldx_model(t, y, ka, ke, Vmax, Km, args):
    ldx_gi, ldx_p, dex_p = y

    efficiency, ke_ldx = args

    conv = (Vmax * ldx_p) / (Km + ldx_p)
    
    ldx_gi_dt = -ka * ldx_gi
    ldx_p_dt = ka * ldx_gi - conv - ke_ldx * ldx_p
    dex_p_dt = efficiency * conv - (ke * dex_p)
        
    return (ldx_gi_dt, ldx_p_dt, dex_p_dt)

def optimize_ldx(kinetics: KineticsFromProDrug, model_func, t_span: tuple[float, float], dose_ug: float,
                   optimize_start: Sequence[float, float]):
    result = optimize_michaelis_menten_kinetics(
        optimize_start,
        model_func,
        t_span,
        kinetics.Vd,
        kinetics.ke,
        kinetics.ka,        
        dose_ug,
        kinetics.Tmax,
        kinetics.Cmax,
        kinetics.auc,
        kinetics.efficiency,
        kinetics.ke_pro
    )
    return result

def calculate_curve(model_func, kinetics: KineticsFromProDrug, t_start: float, t_end: float, dose_mg: float,
                    y0: Sequence[float] = None) -> tuple:
    if y0 is None:
        y0 = [dose_mg*1000, 0, 0]
        
    t_span = (t_start, t_end)

    solution = solve_odes_michaelis(
        model_func, t_span, y0, 
        kinetics.ka, 
        kinetics.ke, 
        kinetics.Vmax, 
        kinetics.Km,
        kinetics.efficiency, kinetics.ke_pro)
    
    t = solution.t
    y = solution.y
    
    Cdex_ng = (y[2] / kinetics.Vd) * 1000

    return (y, t, Cdex_ng)

def simulate(model_func: callable, kinetics: KineticsFromProDrug, t_end: float, doses_mg: float | tuple, times_btwn_doses: None | list):
    if times_btwn_doses is None:
        return calculate_curve(model_func, kinetics, 0, t_end, doses_mg)
    else:
        if type(doses_mg) is tuple:
            return simulate_dif_doses(model_func, kinetics, t_end, doses_mg, times_btwn_doses)
        else:
            return simulate_mult_doses(model_func, kinetics, t_end, doses_mg, times_btwn_doses)

def simulate_mult_doses(model_func: callable, kinetics: KineticsFromProDrug, t_end: float, dose_mg: float, times_btwn_doses: list):
    times_btwn_doses.append(t_end)
    y, t, ng = calculate_curve(model_func, kinetics, 0, times_btwn_doses[0], dose_mg)
    y_all, t_all, ng_all = y, t, ng
    t_next = times_btwn_doses[0]
    for i in range(len(times_btwn_doses)-1):

        dose_ug = dose_mg*1000
        y, t, ng = calculate_curve(model_func, kinetics, t_next, t_next+times_btwn_doses[i+1], dose_mg,
                                y0=[y[0][-1]+dose_ug, y[1][-1], y[2][-1]])
        
        t_next = t_next + times_btwn_doses[i+1]
        y_all = np.concatenate([y_all, y])
        t_all = np.concatenate([t_all, t], axis=None)
        ng_all = np.concatenate([ng_all, ng], axis=None)
    
    return (y_all, t_all, ng_all)

def simulate_dif_doses(model_func: callable, kinetics: KineticsFromProDrug, t_end: float, doses: list, times_btwn_doses: list):
    times_btwn_doses.append(t_end)

    doses_que = deque(doses)

    dose = doses_que.popleft()

    y, t, ng = calculate_curve(model_func, kinetics, 0, times_btwn_doses[0], dose)
    y_all, t_all, ng_all = y, t, ng

    t_next = times_btwn_doses[0]

    for i in range(len(times_btwn_doses)-1):

        try:
            dose = doses_que.popleft()
        except IndexError:
            pass
        
        dose_ug = dose*1000
        y, t, ng = calculate_curve(model_func, kinetics, t_next, t_next+times_btwn_doses[i+1], dose, y0=[y[0][-1]+dose_ug, y[1][-1], y[2][-1]] )

        t_next = t_next + times_btwn_doses[i+1]

        y_all = np.concatenate([y_all, y])
        t_all = np.concatenate([t_all, t], axis=None)
        ng_all = np.concatenate([ng_all, ng], axis=None)
    
    return (y_all, t_all, ng_all)