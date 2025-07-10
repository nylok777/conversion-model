from kinetics import KineticsMM, KineticsFO
from ldx import ldx_model, get_result_ldx, test_plot, calculate_curve
import firstOrder

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

kinetics_alp = KineticsFO(11, 50_000, 1.9, 33, 510)
y_alp, t_alp, Calp_ng = firstOrder.calculate_curve(firstOrder.first_order_model, kinetics_alp, 0, 60, 2)
firstOrder.test_plot(t_alp, Calp_ng, 2, kinetics_alp.Tmax, kinetics_alp.Cmax)