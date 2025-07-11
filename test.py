from kinetics import KineticsFromProDrug, KineticsFO
from ldx import ldx_model, optimize_ldx, test_plot, calculate_curve
import firstOrder as fo

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

y, t, Cdex_ng = calculate_curve(ldx_model, kinetics, 0, 96, 50)

test_plot(t, Cdex_ng, 50, kinetics.Tmax, kinetics.Cmax)

kinetics_alp = KineticsFO(
    t_half=11,
    Vd=50_000,
    Tmax=1.9,
    bioaval=0.9,
    Cmax=33,
    auc=510
)

y_alp, t_alp, Calp_ng = fo.calculate_curve(fo.first_order_model, kinetics_alp, 0, 60, 2)
fo.test_plot(t_alp, Calp_ng, 2, kinetics_alp.Tmax, kinetics_alp.Cmax)