import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import math
import pandas as pd

# --- DATASET ---
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
    """Retrieves fluid properties using CoolProp."""
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
    """Calculates mass flow rate using the Dyer NHNE model for a single point."""
    props1 = get_fluid_properties(fluid, T1_c, P1_bar)
    rho1, h1, s1, p_sat_bar = props1['rho'], props1['h'], props1['s'], props1['P_vap_bar']
    
    delta_p_bar = P1_bar - P2_bar
    if delta_p_bar <= 0: return 0.0
    
    G_spi_ideal = math.sqrt(2 * rho1 * (delta_p_bar * 100000))
    
    if P2_bar >= p_sat_bar: # Subcooled or saturated liquid flow
        mdot = Cd * G_spi_ideal * (num_ports * (math.pi * ((D_mm/1000.0)**2) / 4.0))
        return mdot
    
    # Two-phase flow calculation
    try:
        rho2_hem = CP.PropsSI('D', 'P', P2_bar * 100000, 'S', s1, fluid)
        h2_hem = CP.PropsSI('H', 'P', P2_bar * 100000, 'S', s1, fluid)
        G_hem_ideal = rho2_hem * math.sqrt(2 * (h1 - h2_hem)) if h1 >= h2_hem else 0
    except ValueError:
        G_hem_ideal = 0.0
        
    kappa = math.sqrt((P1_bar - P2_bar) / (p_sat_bar - P2_bar)) if p_sat_bar > P2_bar else float('inf')
    
    if math.isinf(kappa): # Avoid division by zero if P_sat equals P2
        return 0.0

    G_dyer_ideal = (kappa / (1 + kappa)) * G_spi_ideal + (1 / (1 + kappa)) * G_hem_ideal
    A_total = num_ports * (math.pi * ((D_mm/1000.0)**2) / 4.0)
    mdot_final = Cd * G_dyer_ideal * A_total
    return mdot_final

def get_dp_dt_sat(T_k, fluid):
    """Calculates the slope of the saturation pressure curve."""
    P_plus = CP.PropsSI('P', 'T', T_k + 0.01, 'Q', 0, fluid)
    P_minus = CP.PropsSI('P', 'T', T_k - 0.01, 'Q', 0, fluid)
    return (P_plus - P_minus) / 0.02

def calculate_omega_sat(T_sat_k, fluid):
    """Calculates the dimensionless property omega."""
    h_lg = CP.PropsSI('H', 'T', T_sat_k, 'Q', 1, fluid) - CP.PropsSI('H', 'T', T_sat_k, 'Q', 0, fluid)
    c_pl = CP.PropsSI('C', 'T', T_sat_k, 'Q', 0, fluid)
    rho_l = CP.PropsSI('D', 'T', T_sat_k, 'Q', 0, fluid)
    v_l = 1.0 / rho_l
    dp_dt = get_dp_dt_sat(T_sat_k, fluid)
    omega_sat = (c_pl * T_sat_k * v_l / ((h_lg/1000)**2)) * (dp_dt/1000)
    return omega_sat

def get_supercharging_state(P1_bar, T1_c, fluid):
    """Determines if the flow is in a high or low supercharging state."""
    T_sat_k = T1_c + 273.15
    P_sat_bar = CP.PropsSI('P', 'T', T_sat_k, 'Q', 0, fluid) / 1e5
    omega_sat = calculate_omega_sat(T_sat_k, fluid)
    eta_st = (2 * omega_sat) / (1 + 2 * omega_sat)
    state = 'high' if P_sat_bar < (eta_st * P1_bar) else 'low'
    return state, omega_sat, eta_st

def get_low_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid, omega_sat):
    """Calculates choked mass flow for the low supercharge case."""
    T1_k, P1_pa = T1_c + 273.15, P1_bar * 1e5
    P_sat_pa = CP.PropsSI('P', 'T', T1_k, 'Q', 0, fluid)
    rho1 = CP.PropsSI('D', 'P', P1_pa, 'T', T1_k, fluid)
    v1 = 1.0 / rho1
    eta_sat = P_sat_pa / P1_pa
    max_G, P_crit_pa = 0.0, 0.0
    
    for P2_pa_test in np.linspace(P1_pa, 1e5, 1500):
        eta = P2_pa_test / P1_pa
        if not (0 < eta < eta_sat): continue
        
        num_term_inside_sqrt = 2*(1-eta_sat) + 2*(omega_sat*eta_sat*math.log(eta_sat/eta) - (omega_sat-1)*(eta_sat-eta))
        if num_term_inside_sqrt < 0: continue
        
        numerator = math.sqrt(num_term_inside_sqrt)
        denominator = omega_sat * (eta_sat / eta - 1) + 1
        if abs(denominator) < 1e-9: continue
        
        G_low_ideal = (numerator / denominator) * math.sqrt(P1_pa / v1)
        if G_low_ideal > max_G:
            max_G, P_crit_pa = G_low_ideal, P2_pa_test
            
    if max_G == 0.0: return None, None
    
    P_crit_bar = P_crit_pa / 1e5
    A_total = num_ports * (math.pi * ((D_mm / 1000.0)**2) / 4.0)
    mdot_choked = Cd * max_G * A_total
    return mdot_choked, P_crit_bar

def get_high_supercharge_mdot(P1_bar, T1_c, D_mm, num_ports, Cd, fluid):
    """Calculates choked mass flow for the high supercharge case."""
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
    """Calculates mass flow rate using the full Nino & Razavi model."""
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
        # For unchoked flow, it reverts to a two-phase model like Dyer's
        return get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid)

# ---------- PLOTTING AND SIMULATION FUNCTIONS ----------
def run_models_vs_dp(P1_input, P1_unit, T1_input, T1_unit, D_mm, Cd, num_ports, dp_unit, fluid="N2O"):
    """Runs the models against the experimental data and plots results."""
    # Convert inputs to standard units (bar, C)
    P1_bar = float(P1_input) / 14.5038 if P1_unit == "psi" else float(P1_input)
    T1_c = float(T1_input) if T1_unit.lower() == "c" else float(T1_input) - 273.15

    # Prepare experimental data based on chosen unit
    if dp_unit == "psi":
        waxman_exp_data = waxman_exp_data_psi
    else: # bar
        waxman_exp_data = [(item[0] / 14.5038, item[1]) for item in waxman_exp_data_psi]
        
    delta_p_exp = np.array([item[0] for item in waxman_exp_data])
    mdot_exp_kgs = np.array([item[1] for item in waxman_exp_data])

    # Calculate model predictions
    mdot_dyer, mdot_nino = [], []
    for dp in delta_p_exp:
        delta_p_bar = dp / 14.5038 if dp_unit == "psi" else dp
        P2_bar = P1_bar - delta_p_bar
        mdot_dyer.append(get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid))
        mdot_nino.append(get_nino_razavi_mdot(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid))
    
    mdot_dyer, mdot_nino = np.array(mdot_dyer), np.array(mdot_nino)

    # Calculate errors
    percent_error_dyer = np.abs((mdot_dyer - mdot_exp_kgs) / mdot_exp_kgs) * 100
    percent_error_nino = np.abs((mdot_nino - mdot_exp_kgs) / mdot_exp_kgs) * 100
    mape_dyer, mape_nino = np.mean(percent_error_dyer), np.mean(percent_error_nino)

    # Plot 1: Mass Flow Rate vs. Experiment
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

    # Plot 2: Percent Error vs. Experiment
    plt.figure(figsize=(12, 4))
    plt.plot(delta_p_exp, percent_error_nino, 'r-^', label='Nino & Razavi Model')
    plt.plot(delta_p_exp, percent_error_dyer, 'g-s', label='Dyer NHNE Model')
    plt.axhline(y=mape_nino, color='r', linestyle='--', label=f'Nino & Razavi MAPE: {mape_nino:.2f}%')
    plt.axhline(y=mape_dyer, color='g', linestyle='--', label=f'Dyer NHNE MAPE: {mape_dyer:.2f}%')
    plt.title('Percent Error of Mass Flow Rate Prediction vs Experiment', fontsize=15)
    plt.xlabel(f'Injector ΔP ({dp_unit})', fontsize=14)
    plt.ylabel('Percent Error (%)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Print summary table
    summary = pd.DataFrame({
        'Model': ['Nino & Razavi (2019) Proposed', 'Dyer (NHNE) (Waxman/Dyer)'],
        'MAPE (%)': [mape_nino, mape_dyer]
    })
    print("\n--- SUMMARY TABLE ---")
    print(summary)

def run_time_plot(P1_bar, T1_c, P2_bar, D_mm, Cd, num_ports, duration_s, model_choice, fluid="N2O"):
    """
    Calculates and plots mass flow and pressures over time for constant P1, T1, and P2.
    """
    # Calculate the constant mass flow rate using the chosen model
    if model_choice == "Nino & Razavi":
        mdot = get_nino_razavi_mdot(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid)
    elif model_choice == "Dyer":
        mdot = get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, num_ports, Cd, fluid)
    else:
        messagebox.showerror("Error", "Invalid model selected.")
        return

    if mdot <= 0:
        messagebox.showinfo("Result", "Mass flow rate is zero or negative. No plot will be generated.")
        return

    # Create time and data arrays
    time_values = np.linspace(0, duration_s, 500)
    mdot_values = np.full_like(time_values, mdot)
    total_mass_values = mdot * time_values
    p1_values = np.full_like(time_values, P1_bar)
    p2_values = np.full_like(time_values, P2_bar)
    delta_p_values = np.full_like(time_values, P1_bar - P2_bar)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Time Simulation ({model_choice} Model)\nP1={P1_bar:.2f} bar, T1={T1_c:.2f}°C, P2={P2_bar:.2f} bar', fontsize=16)

    # Subplot 1: Mass Flow Rate vs. Time
    ax1.plot(time_values, mdot_values, 'b-')
    ax1.set_ylabel('Mass Flow Rate (kg/s)', fontsize=12)
    ax1.set_title('Instantaneous Mass Flow Rate', fontsize=14)
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.text(0.95, 0.85, f'mdot = {mdot:.4f} kg/s', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Subplot 2: Total Mass Expelled vs. Time
    ax2.plot(time_values, total_mass_values, 'r-')
    ax2.set_ylabel('Total Mass Expelled (kg)', fontsize=12)
    ax2.set_title('Cumulative Mass Expelled', fontsize=14)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    # Subplot 3: Pressures vs. Time
    ax3.plot(time_values, p1_values, 'g-', label=f'Tank Pressure (P1) = {P1_bar:.2f} bar')
    ax3.plot(time_values, p2_values, 'm-', label=f'Chamber Pressure (P2) = {P2_bar:.2f} bar')
    ax3.plot(time_values, delta_p_values, 'k--', label=f'Injector ΔP = {P1_bar - P2_bar:.2f} bar')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Pressure (bar)', fontsize=12)
    ax3.set_title('Pressures vs. Time', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle and x-label
    plt.show()

# ---------- GUI SETUP ----------
root = tk.Tk()
root.title("Hybrid Injector Model GUI")
root.geometry("450x600") # Give the window a default size

# --- Main Input Frame ---
input_frame = tk.LabelFrame(root, text="Model Inputs", padx=10, pady=10)
input_frame.pack(padx=10, pady=10, fill="x")

row = 0
tk.Label(input_frame, text="Upstream Tank Pressure (P1):").grid(row=row, column=0, sticky="w", pady=2)
entry_P1 = tk.Entry(input_frame)
entry_P1.insert(0, "704")
entry_P1.grid(row=row, column=1, sticky="ew")
combo_P1unit = ttk.Combobox(input_frame, values=["psi", "bar"], width=5)
combo_P1unit.set("psi")
combo_P1unit.grid(row=row, column=2, padx=(5,0))
row += 1

temp_label_var = tk.StringVar()
temp_label_var.set("Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]")
label_T1 = tk.Label(input_frame, textvariable=temp_label_var, justify=tk.LEFT)
label_T1.grid(row=row, column=0, sticky="w", pady=2)
entry_T1 = tk.Entry(input_frame)
entry_T1.insert(0, "6.85")
entry_T1.grid(row=row, column=1, sticky="ew")
combo_T1unit = ttk.Combobox(input_frame, values=["C", "K"], width=5)
combo_T1unit.set("C")
combo_T1unit.grid(row=row, column=2, padx=(5,0))
row += 1

def on_T1unit_change(event=None):
    unit = combo_T1unit.get()
    if unit == "C":
        temp_label_var.set("Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]")
        entry_T1.delete(0, tk.END)
        entry_T1.insert(0, "6.85")
    else: # K
        temp_label_var.set("Upstream Tank Temp (T1):\n[N2O: 182.3 to 309.6 K]")
        entry_T1.delete(0, tk.END)
        entry_T1.insert(0, "280")
combo_T1unit.bind("<<ComboboxSelected>>", on_T1unit_change)

tk.Label(input_frame, text="Injector Diameter (mm):").grid(row=row, column=0, sticky="w", pady=2)
entry_D = tk.Entry(input_frame)
entry_D.insert(0, "1.5")
entry_D.grid(row=row, column=1, columnspan=2, sticky="ew")
row += 1

tk.Label(input_frame, text="Discharge Coefficient (Cd):").grid(row=row, column=0, sticky="w", pady=2)
entry_Cd = tk.Entry(input_frame)
entry_Cd.insert(0, "0.77")
entry_Cd.grid(row=row, column=1, columnspan=2, sticky="ew")
row += 1

tk.Label(input_frame, text="Number of Ports:").grid(row=row, column=0, sticky="w", pady=2)
entry_ports = tk.Entry(input_frame)
entry_ports.insert(0, "1")
entry_ports.grid(row=row, column=1, columnspan=2, sticky="ew")
row += 1

tk.Label(input_frame, text="All calculations use N2O.", font=("Helvetica", 8, "italic")).grid(row=row, column=0, columnspan=3, pady=(5,0))
input_frame.columnconfigure(1, weight=1)

# --- ΔP Plot Frame ---
dp_frame = tk.LabelFrame(root, text="ΔP Comparison Plot", padx=10, pady=10)
dp_frame.pack(padx=10, pady=5, fill="x")
tk.Label(dp_frame, text="Experimental ΔP Unit:").pack(side=tk.LEFT, padx=(0, 5))
combo_dpunit = ttk.Combobox(dp_frame, values=["psi", "bar"], width=8)
combo_dpunit.set("psi")
combo_dpunit.pack(side=tk.LEFT)
btn_run_dp = tk.Button(dp_frame, text="Run ΔP Plot", command=lambda: on_run_dp(), bg="#84c784", fg="black")
btn_run_dp.pack(side=tk.RIGHT, padx=(10,0))

# --- Time Plot Frame ---
time_frame = tk.LabelFrame(root, text="Time Simulation", padx=10, pady=10)
time_frame.pack(padx=10, pady=5, fill="x")
time_row = 0

tk.Label(time_frame, text="Calculation Model:").grid(row=time_row, column=0, sticky="w", pady=2)
combo_time_model = ttk.Combobox(time_frame, values=["Nino & Razavi", "Dyer"], width=15)
combo_time_model.set("Nino & Razavi")
combo_time_model.grid(row=time_row, column=1, columnspan=2, sticky="ew")
time_row += 1

tk.Label(time_frame, text="Downstream Pressure (P2):").grid(row=time_row, column=0, sticky="w", pady=2)
entry_P2 = tk.Entry(time_frame)
entry_P2.insert(0, "300")
entry_P2.grid(row=time_row, column=1, sticky="ew")
combo_P2unit = ttk.Combobox(time_frame, values=["psi", "bar"], width=5)
combo_P2unit.set("psi")
combo_P2unit.grid(row=time_row, column=2, padx=(5,0))
time_row += 1

tk.Label(time_frame, text="Simulation Duration (s):").grid(row=time_row, column=0, sticky="w", pady=2)
entry_duration = tk.Entry(time_frame)
entry_duration.insert(0, "10")
entry_duration.grid(row=time_row, column=1, columnspan=2, sticky="ew")
time_row += 1

btn_run_time = tk.Button(time_frame, text="Run Time Plot", command=lambda: on_run_time(), bg="#84a9c7", fg="black")
btn_run_time.grid(row=time_row, column=0, columnspan=3, pady=(5,0))
time_frame.columnconfigure(1, weight=1)

# ---------- GUI ACTION HANDLERS ----------
def get_common_inputs():
    """Helper function to get and validate common inputs from the GUI."""
    P1_input = float(entry_P1.get())
    P1_unit = combo_P1unit.get()
    T1_input = float(entry_T1.get())
    T1_unit = combo_T1unit.get()
    D = float(entry_D.get())
    Cd = float(entry_Cd.get())
    ports = int(entry_ports.get())
    fluid = "N2O"

    # Temperature validation
    if T1_unit.upper() == "C":
        if not (-90 <= T1_input <= 36):
            messagebox.showerror("Input Error", f"For Celsius, enter a value between -90 and 36°C for N₂O.")
            return None
    else: # Kelvin
        if not (182.3 <= T1_input <= 309.6):
            messagebox.showerror("Input Error", f"For Kelvin, enter a value between 182.3 and 309.6 K for N₂O.")
            return None
    
    return P1_input, P1_unit, T1_input, T1_unit, D, Cd, ports, fluid

def on_run_dp():
    """Handler for the 'Run ΔP Plot' button."""
    try:
        inputs = get_common_inputs()
        if inputs is None: return
        P1, P1_unit, T1, T1_unit, D, Cd, ports, fluid = inputs
        dp_unit = combo_dpunit.get()
        run_models_vs_dp(P1, P1_unit, T1, T1_unit, D, Cd, ports, dp_unit, fluid)
    except Exception as e:
        messagebox.showerror("Input Error", f"Check your inputs!\n\nError: {e}")

def on_run_time():
    """Handler for the 'Run Time Plot' button."""
    try:
        inputs = get_common_inputs()
        if inputs is None: return
        P1_input, P1_unit, T1_input, T1_unit, D, Cd, ports, fluid = inputs

        # Get time simulation specific inputs
        model_choice = combo_time_model.get()
        P2_input = float(entry_P2.get())
        P2_unit = combo_P2unit.get()
        duration = float(entry_duration.get())

        # Convert inputs to standard units (bar, C)
        P1_bar = P1_input / 14.5038 if P1_unit == "psi" else P1_input
        P2_bar = P2_input / 14.5038 if P2_unit == "psi" else P2_input
        T1_c = T1_input if T1_unit.upper() == "C" else T1_input - 273.15
        
        if P1_bar <= P2_bar:
            messagebox.showerror("Input Error", "Upstream pressure (P1) must be greater than downstream pressure (P2).")
            return

        run_time_plot(P1_bar, T1_c, P2_bar, D, Cd, ports, duration, model_choice, fluid)
    except Exception as e:
        messagebox.showerror("Input Error", f"Check your inputs!\n\nError: {e}")

root.mainloop()
