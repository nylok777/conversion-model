from collections.abc import Sequence
import ast
import plotting

def get_user_input(substance: str) -> tuple:
    mult_dose_input = input("multiple doses(y/n): ")
    multiple_dose = True if mult_dose_input == 'y' else False
    if multiple_dose:    
        times_btwn_doses = list(map(float, input("time between doses, separated by commas (hours): ").split(',')))
        doses_mg = input(f"doses of {substance} in mg (if different amounts seperate by comma): ")
    else:
        times_btwn_doses = None
        doses_mg = input(f"dose of {substance} (mg): ")
    doses_mg = ast.literal_eval(doses_mg)
    t_end = float(input("length of simulation in hours: "))

    return (t_end, doses_mg, times_btwn_doses, multiple_dose)

def ldx_plot():
    from kinetics import KineticsFromProDrug
    from ldx import ldx_model, optimize_ldx, simulate, calculate_curve

    substance = {'active': 'd-Amphetamine', 'prodrug': 'LDX'}

    """
    Data:
    Ermer, J.C., Pennick, M. & Frick, G. Lisdexamfetamine Dimesylate: Prodrug Delivery, Amphetamine Exposure and Duration of Efficacy. Clin Drug Investig 36, 341–356 (2016). https://doi.org/10.1007/s40261-015-0354-y
    """

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

    sim_length, doses, times_list, multiple_dose = get_user_input("LDX")
    
    if multiple_dose:
        last_dose = input("plot last dose only [y/n]: ")

        if last_dose == 'y':
            sim_length = 24            
            results = simulate(ldx_model, kinetics, times_list[-1], doses, times_list[:-1])
            y, t, conc_ng = results

            if type(doses) is tuple:
                dose_ug = doses[-1]*1000
            else:
                dose_ug = doses*1000
                
            y0 = [y[0][-1]+dose_ug, y[1][-1], y[2][-1]]

            plotting.plot_last_dose(calculate_curve, ldx_model, y0, doses, substance, sim_length, kinetics)
        else:
            results = simulate(ldx_model, kinetics, sim_length, doses, times_list)
            y, t, conc_ng = results
            plotting.draw_full_plot(t, conc_ng, doses, substance, sim_length)

    else:
        results = simulate(ldx_model, kinetics, sim_length, doses, times_list)
        y, t, conc_ng = results
        plotting.draw_full_plot(t, conc_ng, doses, substance, sim_length)

def flvx_plot():
    from kinetics import KineticsFO
    from firstOrder import first_order_model, plot_last_dose

    flvx = KineticsFO(
        t_half=12.5,
        Vd=25_000,
        Tmax=6,
        Cmax=41.88,
        auc=959.33
    )

    plot_last_dose(first_order_model, 50, 24, flvx, "Fluvoxamine")

if __name__ == '__main__':
    ldx_plot()