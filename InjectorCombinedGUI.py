import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import math
import pandas as pd

# Waxman dataset (psi, kg/s)
waxman_exp_data_psi = [
    (5, 0.007), (10, 0.011), (15, 0.014), (20, 0.017), (25, 0.020),
    (30, 0.023), (40, 0.028), (50, 0.031), (60, 0.035), (70, 0.039),
    (80, 0.043), (90, 0.046), (100, 0.049), (110, 0.052), (120, 0.055),
    (130, 0.057), (140, 0.059), (150, 0.061), (160, 0.062), (170, 0.064),
    (180, 0.065), (190, 0.0655), (200, 0.066), (210, 0.0665), (220, 0.067),
    (230, 0.067), (240, 0.0673), (250, 0.0675), (260, 0.0675), (270, 0.0675),
    (280, 0.0675), (290, 0.0675), (300, 0.0675), (310, 0.0674), (320, 0.0673),
    (330, 0.0672), (340, 0.0671), (350, 0.067), (360, 0.0669), (370, 0.0668),
    (380, 0.0667), (390, 0.0666), (400, 0.0665), (410, 0.0663), (420, 0.066)
]

# ---------- MODEL FUNCTIONS ----------
def get_fluid_properties(fluid, T_c, P_bar):
    T_k = T_c + 273.15
    P_pa = P_bar * 100000
    props = {
        'rho': CP.PropsSI('D', 'T', T_k, 'P', P_pa, fluid),
        'h': CP.PropsSI('H', 'T', T_k, 'P', P_pa, fluid),
        's': CP.PropsSI('S', 'T', T_k, 'P', P_pa, fluid),
        'P_vap_bar': CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 100000
    }
    return props

def get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid):
    props1 = get_fluid_properties(fluid, T1_c, P1_bar)
    rho1, h1, s1, p_sat_bar = props1['rho'], props1['h'], props1['s'], props1['P_vap_bar']
    delta_p_bar = P1_bar - P2_bar
    if delta_p_bar <= 0: return 0.0
    G_spi_ideal = math.sqrt(2 * rho1 * (delta_p_bar * 100000))
    if P2_bar >= p_sat_bar:
        mdot = Cd * G_spi_ideal * (num_ports * (math.pi * ((D_mm/1000.0)**2) / 4.0))
        return mdot
    try:
        rho2_hem = CP.PropsSI('D', 'P', P2_bar * 100000, 'S', s1, fluid)
        h2_hem = CP.PropsSI('H', 'P', P2_bar * 100000, 'S', s1, fluid)
        G_hem_ideal = rho2_hem * math.sqrt(2 * (h1 - h2_hem)) if h1 >= h2_hem else 0
    except ValueError:
        G_hem_ideal = 0.0
    kappa = math.sqrt((P1_bar - P2_bar) / (p_sat_bar - P2_bar))
    G_dyer_ideal = (kappa / (1 + kappa)) * G_spi_ideal + (1 / (1 + kappa)) * G_hem_ideal
    A_total = num_ports * (math.pi * ((D_mm/1000.0)**2) / 4.0)
    mdot_final = Cd * G_dyer_ideal * A_total
    return mdot_final

def get_dp_dt_sat(T_k, fluid):
    P_plus = CP.PropsSI('P', 'T', T_k + 0.01, 'Q', 0, fluid)
    P_minus = CP.PropsSI('P', 'T', T_k - 0.01, 'Q', 0, fluid)
    return (P_plus - P_minus) / 0.02

def calculate_omega(P1_bar, T1_c, fluid):
    T1_k = T1_c + 273.15
    h_lg = CP.PropsSI('H', 'T', T1_k, 'Q', 1, fluid) - CP.PropsSI('H', 'T', T1_k, 'Q', 0, fluid)
    c_pl = CP.PropsSI('C', 'T', T1_k, 'Q', 0, fluid)
    rho_l = CP.PropsSI('D', 'T', T1_k, 'Q', 0, fluid)
    v_l = 1.0 / rho_l
    dp_dt = get_dp_dt_sat(T1_k, fluid)
    omega = (c_pl * T1_k * v_l / (h_lg**2)) * dp_dt
    return omega

def get_supercharging_state(P1_bar, T1_c, fluid):
    P_sat_bar = CP.PropsSI('P', 'T', T1_c + 273.15, 'Q', 0, fluid) / 100000
    omega_sat = calculate_omega(P_sat_bar, T1_c, fluid)
    eta_st = (2 * omega_sat) / (1 + 2 * omega_sat)
    state = 'high' if P_sat_bar < (eta_st * P1_bar) else 'low'
    return state, omega_sat, eta_st

def get_low_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid, omega_sat):
    T1_k, P1_pa = T1_c + 273.15, P1_bar * 1e5
    P_sat_pa = CP.PropsSI('P', 'T', T1_k, 'Q', 0, fluid)
    rho1 = CP.PropsSI('D', 'P', P1_pa, 'T', T1_k, fluid)
    v1 = 1.0 / rho1
    eta_sat = P_sat_pa / P1_pa
    max_G, P_crit_pa = 0.0, 0.0
    for P2_pa_test in np.linspace(P1_pa, 1e5, 1500):
        eta = P2_pa_test / P1_pa
        if eta <= 0 or eta >= eta_sat: continue
        num_term_inside_sqrt = 2*(1-eta_sat) + 2*(omega_sat*eta_sat*math.log(eta_sat/eta) - (omega_sat-1)*(eta_sat-eta))
        if num_term_inside_sqrt < 0: continue
        numerator = math.sqrt(num_term_inside_sqrt)
        denominator = omega_sat * (eta_sat / eta - 1) + 1
        if denominator <= 1e-9: continue
        G_low_ideal = (numerator / denominator) * math.sqrt(P1_pa / v1)
        if G_low_ideal > max_G:
            max_G, P_crit_pa = G_low_ideal, P2_pa_test
    if max_G == 0.0: return None, None
    P_crit_bar = P_crit_pa / 1e5
    A_total = num_ports * (math.pi * ((D_mm / 1000.0)**2) / 4.0)
    mdot_choked = Cd * max_G * A_total
    return mdot_choked, P_crit_bar

def get_high_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid):
    T1_k, P1_pa = T1_c + 273.15, P1_bar * 1e5
    P_sat_pa = CP.PropsSI('P', 'T', T1_k, 'Q', 0, fluid)
    rho1 = CP.PropsSI('D', 'P', P1_pa, 'T', T1_k, fluid)
    P_crit_bar = P_sat_pa / 1e5
    if P1_pa < P_sat_pa: return 0.0, P_crit_bar
    G_high_crit_ideal = math.sqrt(2 * rho1 * (P1_pa - P_sat_pa))
    A_total = num_ports * (math.pi * ((D_mm / 1000.0)**2) / 4.0)
    mdot_choked = Cd * G_high_crit_ideal * A_total
    return mdot_choked, P_crit_bar

def get_nino_razavi_mdot(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid):
    state, omega_sat, _ = get_supercharging_state(P1_bar, T1_c, fluid)
    if state == 'high':
        mdot_choked, P_crit_bar = get_high_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid)
    else:
        mdot_choked, P_crit_bar = get_low_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid, omega_sat)
    if mdot_choked is None:
        return 0.0
    if P2_bar <= P_crit_bar:
        return mdot_choked
    else:
        return get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid)

def run_models(P1_input, P1_unit, T1_input, T1_unit, D_mm, Cd, num_ports, dp_unit, fluid="N2O"):
    if P1_unit == "psi":
        P1_bar = float(P1_input) / 14.5038
    else:
        P1_bar = float(P1_input)
    if T1_unit.lower() == "c":
        T1_c = float(T1_input)
        T1_K = T1_c + 273.15
    else:
        T1_K = float(T1_input)
        T1_c = T1_K - 273.15
    if dp_unit == "psi":
        waxman_exp_data = waxman_exp_data_psi
        delta_p_exp = np.array([item[0] for item in waxman_exp_data])
    else:
        waxman_exp_data = [(item[0] / 14.5038, item[1]) for item in waxman_exp_data_psi]
        delta_p_exp = np.array([item[0] for item in waxman_exp_data])
    mdot_exp_kgs = np.array([item[1] for item in waxman_exp_data])
    mdot_dyer, mdot_nino = [], []
    for dp in delta_p_exp:
        delta_p_bar = dp / 14.5038 if dp_unit == "psi" else dp
        P2_bar = P1_bar - delta_p_bar
        mdot_dyer.append(get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid))
        mdot_nino.append(get_nino_razavi_mdot(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid))
    mdot_dyer, mdot_nino = np.array(mdot_dyer), np.array(mdot_nino)
    percent_error_dyer = np.abs((mdot_dyer - mdot_exp_kgs) / mdot_exp_kgs) * 100
    percent_error_nino = np.abs((mdot_nino - mdot_exp_kgs) / mdot_exp_kgs) * 100
    mape_dyer, mape_nino = np.mean(percent_error_dyer), np.mean(percent_error_nino)
    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(delta_p_exp, mdot_exp_kgs, 'ko', label='Experimental Data')
    plt.plot(delta_p_exp, mdot_nino, 'r-^', label=f'Nino & Razavi Model (MAPE: {mape_nino:.2f}%)')
    plt.plot(delta_p_exp, mdot_dyer, 'g-s', label=f'Dyer NHNE Model (MAPE: {mape_dyer:.2f}%)')
    plt.title('Mass Flow Rate Prediction vs Experiment', fontsize=16)
    plt.xlabel(f'Injector ΔP ({dp_unit})', fontsize=14)
    plt.ylabel('Mass Flow Rate (kg/s)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(12, 4))
    # plt.plot(delta_p_exp, percent_error_nino, 'r-^', label='Nino & Razavi Model')
    # plt.plot(delta_p_exp, percent_error_dyer, 'g-s', label='Dyer NHNE Model')
    # plt.axhline(y=mape_nino, color='r', linestyle='--', label=f'Nino & Razavi MAPE: {mape_nino:.2f}%')
    # plt.axhline(y=mape_dyer, color='g', linestyle='--', label=f'Dyer NHNE MAPE: {mape_dyer:.2f}%')
    # plt.title('Percent Error of Mass Flow Rate Prediction vs Experiment', fontsize=15)
    # plt.xlabel(f'Injector ΔP ({dp_unit})', fontsize=14)
    # plt.ylabel('Percent Error (%)', fontsize=14)
    # plt.legend(fontsize=11)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    # plt.show()
    summary = pd.DataFrame({
        'Model': ['Nino & Razavi (2019) Proposed', 'Dyer (NHNE) (Waxman/Dyer)'],
        'MAPE (%)': [mape_nino, mape_dyer]
    })
    print("\n--- SUMMARY TABLE ---")
    print(summary)

# ---------- GUI SETUP ----------
root = tk.Tk()
root.title("Hybrid Injector Model Comparison GUI")
row = 0
tk.Label(root, text="Upstream Tank Pressure (P1):").grid(row=row, column=0, sticky="e")
entry_P1 = tk.Entry(root)
entry_P1.insert(0, "704")
entry_P1.grid(row=row, column=1)
combo_P1unit = ttk.Combobox(root, values=["psi", "bar"], width=5)
combo_P1unit.set("psi")
combo_P1unit.grid(row=row, column=2)
row += 1

# Temperature label as variable
temp_label_var = tk.StringVar()
temp_label_var.set("Upstream Tank Temperature (T1):\n[N2O: -90°C–36°C]")
label_T1 = tk.Label(root, textvariable=temp_label_var)
label_T1.grid(row=row, column=0, sticky="e")
entry_T1 = tk.Entry(root)
entry_T1.insert(0, "6.85")
entry_T1.grid(row=row, column=1)
combo_T1unit = ttk.Combobox(root, values=["C", "K"], width=5)
combo_T1unit.set("C")
combo_T1unit.grid(row=row, column=2)
row += 1

def on_T1unit_change(event=None):
    unit = combo_T1unit.get()
    if unit == "C":
        temp_label_var.set("Upstream Tank Temperature (T1):\n[N2O: -90°C–36°C]")
        entry_T1.delete(0, tk.END)
        entry_T1.insert(0, "6.85")
    else:
        temp_label_var.set("Upstream Tank Temperature (T1):\n[N2O: 182.3–309.6 K]")
        entry_T1.delete(0, tk.END)
        entry_T1.insert(0, "280")
combo_T1unit.bind("<<ComboboxSelected>>", on_T1unit_change)

tk.Label(root, text="Injector Diameter (mm):").grid(row=row, column=0, sticky="e")
entry_D = tk.Entry(root)
entry_D.insert(0, "1.5")
entry_D.grid(row=row, column=1)
row += 1

tk.Label(root, text="Discharge Coefficient (Cd):").grid(row=row, column=0, sticky="e")
entry_Cd = tk.Entry(root)
entry_Cd.insert(0, "0.77")
entry_Cd.grid(row=row, column=1)
row += 1

tk.Label(root, text="Number of Ports:").grid(row=row, column=0, sticky="e")
entry_ports = tk.Entry(root)
entry_ports.insert(0, "1")
entry_ports.grid(row=row, column=1)
row += 1

tk.Label(root, text="Experimental ΔP Unit:").grid(row=row, column=0, sticky="e")
combo_dpunit = ttk.Combobox(root, values=["psi", "bar"], width=5)
combo_dpunit.set("psi")
combo_dpunit.grid(row=row, column=1)
row += 1

def on_run():
    try:
        P1 = float(entry_P1.get())
        T1 = float(entry_T1.get())
        D = float(entry_D.get())
        Cd = float(entry_Cd.get())
        ports = int(entry_ports.get())
        fluid = "N2O"
        t_unit = combo_T1unit.get().upper()
        # Convert to Kelvin for validation
        if t_unit == "C":
            T1_K = T1 + 273.15
            if not (-90 <= T1 <= 36):
                messagebox.showerror("Input Error", f"For Celsius, enter a value between -90 and 36°C for N₂O.")
                return
        else:
            T1_K = T1
            if not (182.3 <= T1_K <= 309.6):
                messagebox.showerror("Input Error", f"For Kelvin, enter a value between 182.3 and 309.6 K for N₂O.")
                return
        run_models(P1, combo_P1unit.get(), T1, combo_T1unit.get(), D, Cd, ports, combo_dpunit.get(), fluid)
    except Exception as e:
        messagebox.showerror("Input Error", f"Check your inputs!\n\nError: {e}")

tk.Button(root, text="Run", command=on_run, bg="#84c784", fg="black", height=2, width=12).grid(row=row, column=0, columnspan=3, pady=10)
row += 1

tk.Label(root, text="All calculations use N2O.").grid(row=row, column=0, columnspan=3)
row += 1

root.mainloop()
