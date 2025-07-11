def ldx_plot():
    from kinetics import KineticsFromProDrug
    from ldx import ldx_model, optimize_ldx, simulate, plot_last_dose

    kinetics = KineticsFromProDrug(
        0.297,
        0.9,
        1.15,
        11.3,
        195_000,
        4,
        44.6,
        763.1
    )

    kinetics.get_michaelis_params(optimize_ldx, ldx_model, (0, 96), 50_000, (25_000, 20_000))

    full = [22, 28.84, 24.083, 23.684, 23.54, 24.116, 21.384, 24.6, 28.216, 22.25, 25.6, 23.83, 21.083, 23.083, 49, 27.734]

    results = simulate(ldx_model, kinetics, full[-1], 30, full[:-2])
    y = results[0]
    y0 = [30_000, y[-2][-1], y[-1][-1]]
    plot_last_dose(y0, 30, 10, kinetics)

def flvx_plot():
    from kinetics import KineticsFO
    from firstOrder import first_order_model, plot_last_dose

    flvx = KineticsFO(
        t_half=12.5,
        Vd=25_000,
        Tmax=6,
        bioaval=0.53,
        Cmax=41.88,
        auc=959.33
    )

    plot_last_dose(first_order_model, 50, 24, flvx, "Fluvoxamine")

if __name__ == '__main__':
    flvx_plot()