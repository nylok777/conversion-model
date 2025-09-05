import plotting

from kinetics import KineticsFromProDrug
from ldx import ldx_model, Ldx

substance = {"active": "d-Amphetamine", "prodrug": "LDX"}

"""
Data:
Ermer, J.C., Pennick, M. & Frick, G. Lisdexamfetamine Dimesylate: Prodrug Delivery, Amphetamine Exposure and Duration of Efficacy. Clin Drug Investig 36, 341–356 (2016). https://doi.org/10.1007/s40261-015-0354-y
"""

kinetics = {
    "efficiency": 0.297,
    "t_half_pro": 0.9,
    "Tmax_pro": 1.15,
    "t_half": 11.3,
    "Vd": 195_000,
    "Tmax": 4,
    "Cmax": 44.6,
    "auc": 763.1,
}

ldx = Ldx(ldx_model, **kinetics)

ldx.find_mm_params(t_span=(0.0, 96.0), dose_ug=50_000, initial_guess=(25_000, 20_000))

doses = (50,)
sim_length = 96

results = ldx.simulate(sim_length, doses, None)
if results is not None:
    y, t, conc = results
    plotting.test_plot(t, conc, doses[0], ldx.Tmax, ldx.Cmax)
