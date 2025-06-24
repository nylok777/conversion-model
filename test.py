from kinetics import KineticsMM, Kinetics
from ldx import ldx_model, get_result_ldx, test_plot, calculate_curve
import first_order

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

kinetics_alp = Kinetics(11, 50_000, 0.1666, 4, 0)
y_alp, t_alp, Calp_ng = first_order.calculate_curve(first_order.first_order_model, kinetics_alp, 0, 12, 0.25)
first_order.test_plot(t_alp, Calp_ng, 0.25, kinetics_alp.Tmax, kinetics_alp.Cmax)