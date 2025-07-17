def get_user_input(substance: str) -> tuple:
    multiple_dose = bool(input("multiple doses(yes/no [True/False]): "))
    if multiple_dose:    
        t_doses = list(map(float, input("times between the doses in hours (separated by commas): ")
                              .split(',')))       
    else:
        t_doses = None
    dose_mg = float(input(f"dose of {substance} in mg: "))
    t_end = float(input("length of simulation in hours: "))

    return (t_end, dose_mg, t_doses, multiple_dose)

def ldx_plot():
    from kinetics import KineticsFromProDrug
    from ldx import ldx_model, optimize_ldx, simulate, plot_last_dose, draw_full_plot

    sim_length, dose, times_list, multiple_dose = get_user_input("LDX")
    
    if multiple_dose:
        last_dose = input("plot only last dose [y/n]: ")
        if last_dose.lower() == 'y':
            plot_t_interval = 24
    else:
        plot_t_interval = sim_length

    kinetics = KineticsFromProDrug(
        0.297,
        0.9,
        1.15,
        11.3,
        195_000,
        4,
        1,
        44.6,
        763.1
    )

    kinetics.get_michaelis_params(optimize_ldx, ldx_model, (0, 96), 50_000, (25_000, 20_000))

    results = simulate(ldx_model, kinetics, sim_length, dose, times_list)

    y, t, conc_ng = results

    y0 = [50_000, y[-2][-1], y[-1][-1]]

    if plot_t_interval:
        plot_last_dose(y0, dose, plot_t_interval, kinetics)
    else:
        draw_full_plot(t, conc_ng, 50)    

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
    ldx_plot()