from collections.abc import Sequence
from sklearn.utils import Bunch
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

"""
ka: absorption rate
ke: elimination rate (of active drug)
ke_pro: elimination rate of prodrug
t_span: simulation timespan, for finding Michaelis-Menten parameters should be AUC_0-t
f: conversion efficiency
tmax: point in time when concentration of drug in plasma is max
cmax: max concentration

"""


def objective_michaelis(
        params: tuple[float, float],
        model: callable,
        t_span: Sequence[float, float],
        V: float,
        ka: float,
        ke: float,
        dose_ug: float,
        tmax_target: float, cmax_target: float, auc_target: float,
        args,
        wt: float=10.0, wc: float=1.0, wa: float=1.0, eps=1e-6,
    ) -> float:
    
    Vmax, Km = params
    y0 = (dose_ug, 0, 0)
    
    solution = solve_odes_michaelis(model, t_span, y0, ka, ke, Vmax, Km, *args)
    t = solution.t

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

def solve_odes_michaelis(
        model: callable,
        t_span: Sequence[float, float],
        y0: Sequence[float, float, float],
        ka: float, ke: float,
        Vmax: float, Km: float, *args
    ) -> Bunch:
    t_eval = np.linspace(*t_span, 10_000)

    solution = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(ka, ke, Vmax, Km, args)
    )
    
    return solution

def solve_odes_fo(
        model: callable,
        t_span: Sequence[float, float],
        y0: Sequence[float, float],
        ka: float, ke: float,
        bioaval: float
    ) -> Bunch:
    t_eval = np.linspace(*t_span, 10_000)

    solution = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(ka, ke, bioaval)
    )
    
    return solution

def optimize_michaelis_menten_kinetics(
        initial_guess: Sequence[float, float],
        model: callable,
        t_span: Sequence[float, float],
        V: float,
        ke: float, ka: float,
        dose_ug: float,
        tmax_target: float, cmax_target: float, auc_target:float,
        *args
    ) -> OptimizeResult:
    
    #efficiency, ke_pro = args

    result = minimize(
        fun = objective_michaelis,
        x0 = initial_guess,
        args = (model, t_span, V, ka, ke, dose_ug, tmax_target, cmax_target, auc_target, args),
        method = 'Nelder-Mead'
    )

    return result