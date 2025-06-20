from kinetics import KineticsMM
from ldx import ldx_model, get_result_ldx, test_plot, calculate_curve

kinetics = KineticsMM(
    0.297,
    0.9,
    1.15,
    11.3,
    195_000,
    4,
    44.6,
    763.1
)

kinetics.get_michaelis_params(get_result_ldx, ldx_model, (0, 96), 50_000, (10_000, 5_000))
y, t, Cdex_ng = calculate_curve(ldx_model, kinetics, 0, 96, 50)
test_plot(t, Cdex_ng, 50, kinetics.Tmax, kinetics.Cmax)