import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root_scalar

"""
ka: absorption rate
ke: elimination rate (of active drug)
ke_pro: elimination rate of prodrug
t_span: simulation timespan, for finding Michaelis-Menten parameters should be AUC_0-t
f: conversion efficiency
tmax: point in time when concentration of drug in plasma is max
cmax: max concentration

"""


def objective(params, model, t_span, V, ka, ke, ke_pro, f, dose_ug, tmax_target, cmax_target, auc_target, wt=10.0, wc=1.0, wa=1.0, eps=1e-6):
    
    Vmax, Km = params
    y0 = (dose_ug, 0, 0)
    
    
    solution = solve_odes(model, t_span, y0, ka, ke, ke_pro, f, Vmax, Km)

    t = solution.t
    
    if ke_pro is None:
        active_p = solution.y[1]
    else:
        active_p = solution.y[2]

    C_active = active_p / V
    C_active_ng = C_active * 1000

    idx = np.argmax(C_active_ng)
    tmax = t[idx]
    cmax = C_active_ng[idx]
    auc = np.trapz(C_active_ng, t)

    err_t = (np.log(tmax + eps) - np.log(tmax_target + eps))**2
    err_c = (np.log(cmax + eps) - np.log(cmax_target + eps))**2
    err_a = (np.log(auc + eps) - np.log(auc_target + eps))**2

    return wt * err_t + wc * err_c + wa * err_a

def solve_odes(model, t_span, y0, ka, ke, ke_pro, f, Vmax, Km):
    t_eval = np.linspace(*t_span, 2000)
    solution = solve_ivp(
        model, t_span, y0, t_eval=t_eval, args=(ka, ke, ke_pro, f, Vmax, Km)
    )
    
    return solution

def get_michaelis_menten_params(
        initial_guess, model, 
        t_span: tuple, 
        Vmax_bound: tuple, 
        Km_bound: tuple, 
        V: int | float, 
        ke: int | float,
        ka: float,
        f: float, 
        dose_ug, 
        tmax_target, 
        cmax_target, 
        auc_target, 
        ke_pro: int | float = None):

    result = minimize(
        objective, initial_guess,
        args=(model, t_span, V, ka, ke, ke_pro, f, dose_ug, tmax_target, cmax_target, auc_target),
        bounds=(Vmax_bound, Km_bound),
        method='Nelder-Mead'
    )

    return result