from kinetics import KineticsFromProDrug
from ldx import ldx_model, optimize_ldx, calculate_curve
from plotting import test_plot

kinetics = KineticsFromProDrug(
    efficiency=0.297,
    t_half_pro=0.9,
    Tmax_pro=1.15,
    t_half=11.3,
    Vd=195_000,
    Tmax=4,
    bioaval=0.8,
    Cmax=44.6,
    auc=763.1
)

kinetics.get_michaelis_params(
    result_func=optimize_ldx,
    model_func=ldx_model,
    t_span=(0, 96),
    dose_ug=50_000,
    optimize_start=(10_000, 5_000)
)

dose = 50

y, t, Cdex_ng = calculate_curve(ldx_model, kinetics, 0, 96, dose)

test_plot(t, Cdex_ng, dose, kinetics.Tmax, kinetics.Cmax)