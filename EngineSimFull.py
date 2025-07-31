# --------------------------------------------------------------
# Hybrid-Injector Modeller + Tank Blow-Down GUI
# (c) 2023-25    •    Based on original script by Aadam Awad + Tanin's Modifications
#
# Fully Corrected and Enhanced by Gemini:
# - Integrated the detailed equilibrium blowdown model from BlowdownGeminiDyer.py.
# - Implemented threading for the blowdown calculation to prevent GUI freezing.
# - Added a real-time progress bar and status label for user feedback.
# - Made the GUI fully responsive during long calculations.
# - Added Feed system pressure loss input and calculation.
# - ADDED: Dynamic P2 calculation based on HRAP combustion model.
# - Restored checkbox to select between blowdown and constant pressure simulations.
# - Added inputs for Fuel OD and ID to enable burnout prediction.
# - FIXED: Corrected the unit conversion for regression coefficient 'a' and updated its GUI label.
# --------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import math, sys, traceback
import threading
import queue

sys.setrecursionlimit(50000)

# --- Global variables for parameter passing and results ---
params = {}
result_queue = queue.Queue()

# ---------- DATASET (Waxman 2005) ----------
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

# ---------- A. CORE INJECTOR MODEL FUNCTIONS ----------
def get_subcooled_fluid_properties(fluid, T_c, P_bar):
    T_k, P_pa = T_c + 273.15, P_bar * 1e5
    try:
        return {'rho':CP.PropsSI('D','T',T_k,'P',P_pa,fluid),'h':CP.PropsSI('H','T',T_k,'P',P_pa,fluid),'s':CP.PropsSI('S','T',T_k,'P',P_pa,fluid),'P_vap_bar':CP.PropsSI('P','T',T_k,'Q',0,fluid)/1e5,'T_k':T_k,'P1_bar':P_bar}
    except ValueError: return None

def get_saturated_liquid_properties(fluid, T_c):
    T_k = T_c + 273.15
    try:
        P_sat_pa = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid)
        return {'rho':CP.PropsSI('D','T',T_k,'Q',0,fluid),'h':CP.PropsSI('H','T',T_k,'Q',0,fluid),'s':CP.PropsSI('S','T',T_k,'Q',0,fluid),'P_vap_bar':P_sat_pa/1e5,'T_k':T_k,'P1_bar':P_sat_pa/1e5}
    except ValueError: return None

# --- Nino-Razavi (Ω-model) Helpers ---
def get_dp_dt_sat(T_k, fluid):
    return (CP.PropsSI('P','T',T_k+0.01,'Q',0,fluid) - CP.PropsSI('P','T',T_k-0.01,'Q',0,fluid)) / 0.02

def calculate_omega_sat(T_k, fluid):
    h_lg = CP.PropsSI('H','T',T_k,'Q',1,fluid) - CP.PropsSI('H','T',T_k,'Q',0,fluid)
    c_pl = CP.PropsSI('C','T',T_k,'Q',0,fluid)
    ρ_l  = CP.PropsSI('D','T',T_k,'Q',0,fluid)
    v_l  = 1/ρ_l
    return (c_pl*T_k*v_l / ((h_lg/1000)**2)) * (get_dp_dt_sat(T_k, fluid)/1000)

def get_supercharging_state(P1_bar, T_k, fluid):
    P_sat_bar = CP.PropsSI('P','T',T_k,'Q',0,fluid)/1e5
    ω = calculate_omega_sat(T_k, fluid)
    η_st = (2*ω)/(1+2*ω)
    return ('high' if P_sat_bar < η_st*P1_bar else 'low'), ω

def get_low_supercharge_mdot(props1, D_mm, n_ports, Cd, fluid, ω):
    T_k, P1_pa = props1['T_k'], props1['P1_bar'] * 1e5
    P_sat_pa = props1['P_vap_bar'] * 1e5
    v1 = 1 / props1['rho']
    η_sat = P_sat_pa/P1_pa
    max_G, Pcrit_pa = 0.0, 0.0
    for P2_pa in np.linspace(P1_pa,1e5,1500):
        η = P2_pa/P1_pa
        if not (0<η<η_sat): continue
        ln_term = math.log(η_sat/η)
        inside = 2*(1-η_sat) + 2*(ω*η_sat*ln_term - (ω-1)*(η_sat-η))
        if inside < 0: continue
        denom = ω*(η_sat/η - 1)+1
        if abs(denom)<1e-9: continue
        G_low = (math.sqrt(inside)/denom)*math.sqrt(P1_pa/v1)
        if G_low>max_G:
            max_G, Pcrit_pa = G_low, P2_pa
    if max_G == 0: return None, None
    A_tot = n_ports*math.pi*(D_mm/1000)**2/4
    return Cd*max_G*A_tot, Pcrit_pa/1e5

def get_high_supercharge_mdot(props1, D_mm, n_ports, Cd, fluid):
    P1_pa = props1['P1_bar'] * 1e5
    P_sat_pa = props1['P_vap_bar'] * 1e5
    if P1_pa < P_sat_pa: return 0.0, props1['P_vap_bar']
    G = math.sqrt(2*props1['rho']*(P1_pa-P_sat_pa))
    A_tot = n_ports*math.pi*(D_mm/1000)**2/4
    return Cd*G*A_tot, props1['P_vap_bar']

# --- Unified Mass Flow Calculator ---
def calculate_mdot_from_properties(props1, P2_bar, D_mm, n_ports, Cd, fluid, model_name):
    """Generic mass flow calculator that takes a fluid properties dictionary."""
    if not props1: return 0.0
    ρ1, h1, s1 = props1['rho'], props1['h'], props1['s']
    P1_bar, T_k, P_sat = props1['P1_bar'], props1['T_k'], props1['P_vap_bar']
    A_tot = n_ports * math.pi * (D_mm/1000)**2 / 4
    ΔP_bar = P1_bar - P2_bar
    if ΔP_bar < 0 and model_name != "Nino & Razavi": return 0.0

    # --- SPI Model ---
    if model_name == "SPI (Incompressible)":
        return A_tot * Cd * math.sqrt(2 * ρ1 * ΔP_bar * 1e5)

    # --- Burnell Model ---
    if model_name == "Burnell (Choked)":
        P1_pa = P1_bar * 1e5
        C = -1.5267e-8 * P1_pa + 0.2279
        if C < 0: return 0.0
        return Cd * A_tot * math.sqrt(2 * ρ1 * P1_pa * C)

    # --- Nino-Razavi Model ---
    if model_name == "Nino & Razavi":
        state, ω = get_supercharging_state(P1_bar, T_k, fluid)
        if state == 'high':
            mdot_ch, Pcrit = get_high_supercharge_mdot(props1, D_mm, n_ports, Cd, fluid)
        else:
            mdot_ch, Pcrit = get_low_supercharge_mdot(props1, D_mm, n_ports, Cd, fluid, ω)
        if mdot_ch is None: return 0.0
        # If downstream pressure is above critical pressure, use Dyer model for the unchoked part
        if Pcrit is not None and P2_bar > Pcrit:
             model_name = "Dyer" # Fall through to Dyer logic
        else:
             return mdot_ch

    # --- Dyer NHNE Model (also used as fallback for Nino-Razavi) ---
    G_spi = math.sqrt(2 * ρ1 * ΔP_bar * 1e5)
    if P2_bar >= P_sat:
        return Cd * G_spi * A_tot
    try:
        ρ2 = CP.PropsSI('D', 'P', P2_bar * 1e5, 'S', s1, fluid)
        h2 = CP.PropsSI('H', 'P', P2_bar * 1e5, 'S', s1, fluid)
        G_hem = ρ2 * math.sqrt(max(2 * (h1 - h2), 0))
    except ValueError: G_hem = 0.0

    denominator = P_sat - P2_bar
    if denominator <= 1e-12: κ = float('inf')
    else: κ = math.sqrt((P1_bar - P2_bar) / denominator)

    G_dyer = (κ / (1 + κ)) * G_spi + (1 / (1 + κ)) * G_hem if not math.isinf(κ) else G_spi
    return Cd * G_dyer * A_tot

# ---------- B. SIMULATION SOLVER (THREADED) ----------
def get_initial_tank_conditions_blowdown(m_liq_target, T_c, V_tank, fluid, dV_step=0.0001):
    T_k = T_c + 273.15
    try:
        liquid_density = CP.PropsSI("D", "T", T_k, "Q", 0, fluid)
        max_mass = V_tank * liquid_density
        if m_liq_target > max_mass:
            messagebox.showerror("Input Error", f"Target mass ({m_liq_target:.2f} kg) exceeds max possible mass ({max_mass:.2f} kg).")
            return None
        liquidV = 0; vaporV = V_tank
        while True:
            liquidV += dV_step; vaporV -= dV_step
            if vaporV <= 0: raise RuntimeError("Tank volume too small for target mass.")
            if liquidV * liquid_density >= m_liq_target:
                return {'liq_vol': liquidV, 'vap_vol': vaporV, 'temp_k': T_k}
    except (ValueError, RuntimeError) as e:
        messagebox.showerror("CoolProp/Init Error", f"Could not calculate initial conditions:\n{e}"); return None
    
def run_models_vs_dp(P1_bar, T1_c, D_mm, Cd, n_ports, dp_unit, fluid="N2O"):
    # Note: P1_bar here is already the pressure at the injector inlet
    data = waxman_exp_data_psi if dp_unit=="psi" else [(p/14.5038,m) for p,m in waxman_exp_data_psi]
    ΔP_exp = np.array([d[0] for d in data]); m_exp  = np.array([d[1] for d in data])
    m_dyer, m_nino = [], []
    props1 = get_subcooled_fluid_properties(fluid, T1_c, P1_bar)
    if not props1: messagebox.showerror("Error", "Could not get fluid properties for ΔP plot."); return
    for dp in ΔP_exp:
        P2_bar = P1_bar - (dp / 14.5038 if dp_unit=="psi" else dp)
        m_dyer.append(calculate_mdot_from_properties(props1, P2_bar, D_mm, n_ports, Cd, fluid, "Dyer"))
        m_nino.append(calculate_mdot_from_properties(props1, P2_bar, D_mm, n_ports, Cd, fluid, "Nino & Razavi"))
    m_dyer, m_nino = np.array(m_dyer), np.array(m_nino)
    err_dyer, err_nino = np.abs((m_dyer-m_exp)/m_exp)*100, np.abs((m_nino-m_exp)/m_exp)*100
    MAPE_dyer, MAPE_nino = err_dyer.mean(), err_nino.mean()
    plt.figure(figsize=(12,7)); plt.plot(ΔP_exp,m_exp,'ko',label='Experimental (Waxman 2005)')
    plt.plot(ΔP_exp,m_nino,'r-^',label=f'Nino-Razavi (MAPE {MAPE_nino:.1f}%)')
    plt.plot(ΔP_exp,m_dyer,'g-s',label=f'Dyer (MAPE {MAPE_dyer:.1f}%)')
    plt.title('Mass-Flow vs ΔP',fontsize=16); plt.xlabel(f'ΔP ({dp_unit})'); plt.ylabel('ṁ (kg/s)'); plt.grid(True,ls='--',lw=0.5); plt.legend(); plt.show()
    plt.figure(figsize=(12,4)); plt.plot(ΔP_exp,err_nino,'r-^',label='Nino-Razavi'); plt.plot(ΔP_exp,err_dyer,'g-s',label='Dyer')
    plt.axhline(MAPE_nino,color='r',ls='--'); plt.axhline(MAPE_dyer,color='g',ls='--')
    plt.title('Percent Error'); plt.xlabel(f'ΔP ({dp_unit})'); plt.ylabel('%'); plt.grid(True,ls='--',lw=0.5); plt.legend(); plt.show()


def run_threaded_simulation(local_params, result_queue):
    try:
        is_blowdown = local_params['is_blowdown']
        hrap = local_params['hrap']
        At = math.pi * (hrap['nozzle_throat_diam_mm'] / 1000)**2 / 4
        current_port_diam_m = hrap['initial_port_id_mm'] / 1000
        fuel_od_m = hrap['fuel_od_mm'] / 1000

        if is_blowdown:
            initial_state = get_initial_tank_conditions_blowdown(local_params['initial_mass_kg'], local_params['T_c'], local_params['V_tank_m3'], local_params['fluid'])
            if initial_state is None:
                result_queue.put({'error': "Initialization failed."}); return
            cur_liq_vol, cur_vap_vol, cur_temp = initial_state['liq_vol'], initial_state['vap_vol'], initial_state['temp_k']
            P_comb_bar = (cur_temp / 300) * 30
        else:
            cur_temp = local_params['T_c'] + 273.15
            P_comb_bar = local_params['P1_bar'] * 0.5
            
        time, dt = 0.0, 0.01
        time_vals, mdot_ox_vals, mdot_fuel_vals, p_tank_vals, p_inj_vals, p_comb_vals, of_vals, r_dot_vals = [],[],[],[],[],[],[],[]

        while time < local_params['duration_s']:
            if current_port_diam_m >= fuel_od_m:
                result_queue.put({'warning': f"Fuel burnout detected at t={time:.2f} s. Simulation stopped."})
                break

            if is_blowdown:
                props1_tank = get_saturated_liquid_properties(local_params['fluid'], cur_temp - 273.15)
                if props1_tank is None: break
                P_tank_bar = props1_tank['P1_bar']
            else:
                P_tank_bar = local_params['P1_bar']
                props1_tank = get_subcooled_fluid_properties(local_params['fluid'], cur_temp - 273.15, P_tank_bar)
                if props1_tank is None: break

            P_injector_inlet_bar = P_tank_bar - local_params['pipe_loss_bar']
            if P_injector_inlet_bar <= P_comb_bar: break

            props1_injector = props1_tank.copy(); props1_injector['P1_bar'] = P_injector_inlet_bar
            mDot_ox = calculate_mdot_from_properties(props1_injector, P_comb_bar, local_params['D_mm'], local_params['n_ports'], local_params['Cd'], local_params['fluid'], local_params['model_name'])
            if mDot_ox <= 0: break

            port_area = math.pi * (current_port_diam_m**2) / 4
            G_ox = mDot_ox / port_area
            a_SI = hrap['a'] * 1/1000 # Convert 'a' from mm/s to m/s
            r_dot_m_s = a_SI * (G_ox**hrap['n'])
            mDot_fuel = (math.pi*current_port_diam_m*hrap['grain_length_m'])*r_dot_m_s*hrap['fuel_density_kg_m3']
            mDot_total = mDot_ox + mDot_fuel
            P_comb_bar_new = (mDot_total * hrap['c_star_m_s']) / At / 1e5
            P_comb_bar = 0.6 * P_comb_bar + 0.4 * P_comb_bar_new
            
            time_vals.append(time); mdot_ox_vals.append(mDot_ox); mdot_fuel_vals.append(mDot_fuel); p_tank_vals.append(P_tank_bar); p_inj_vals.append(P_injector_inlet_bar); p_comb_vals.append(P_comb_bar)
            of_vals.append(mDot_ox / mDot_fuel if mDot_fuel > 0 else 0); r_dot_vals.append(r_dot_m_s * 1000)

            if is_blowdown:
                current_liq_dens = props1_tank['rho']
                if cur_liq_vol * current_liq_dens < 0.001: break
                deltaV_expelled = (mDot_ox / current_liq_dens) * dt
                cur_liq_vol -= deltaV_expelled; cur_vap_vol += deltaV_expelled
                try:
                    h_fg = CP.PropsSI("H","T",cur_temp,"Q",1,local_params['fluid']) - CP.PropsSI("H","T",cur_temp,"Q",0,local_params['fluid'])
                    c_p = CP.PropsSI("C", "T", cur_temp, "Q", 0, local_params['fluid'])
                    mass_to_boil = deltaV_expelled * CP.PropsSI("D","T",cur_temp,"Q",1,local_params['fluid'])
                    remaining_liquid_mass = (cur_liq_vol * current_liq_dens)
                    if remaining_liquid_mass > 1e-6: cur_temp -= (mass_to_boil * h_fg) / (remaining_liquid_mass * c_p)
                except ValueError: break
            
            current_port_diam_m += 2 * r_dot_m_s * dt
            time += dt
            result_queue.put({'progress': (time / local_params['duration_s']) * 100})

        if not time_vals:
            result_queue.put({'error': "Simulation did not run. Check initial conditions."}); return

        thrust_N = (np.array(mdot_ox_vals) + np.array(mdot_fuel_vals)) * hrap['c_star_m_s']
        result_queue.put({'results': {
            'time':time_vals, 'mdot_ox':mdot_ox_vals, 'mdot_fuel':mdot_fuel_vals, 'p_tank':p_tank_vals, 'p_inj':p_inj_vals,
            'p_comb':p_comb_vals, 'of':of_vals, 'r_dot':r_dot_vals, 'thrust':thrust_N,
            'summary': {
                "Final Time (s)": time_vals[-1], "Avg. Thrust (N)": np.mean(thrust_N), "Avg. O/F Ratio": np.mean(of_vals),
                "Total Ox Mass (kg)": np.trapezoid(mdot_ox_vals, time_vals), "Total Fuel Mass (kg)": np.trapezoid(mdot_fuel_vals, time_vals),
                "Avg. Chamber Pressure (bar)": np.mean(p_comb_vals),
            }}})
    except Exception as e:
        result_queue.put({'error': f"An error occurred:\n{traceback.format_exc()}"})

# ---------- C. PLOTTING AND ORCHESTRATION ----------
def plot_simulation_results(res, model_name):
    t = res['time']
    if len(t)<2: return
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Full System Simulation ({model_name})', fontsize=18)
    
    axes[0,0].plot(t,res['p_tank'],'g-',label='P_tank'); axes[0,0].plot(t,res['p_inj'],'m-.',label='P_injector'); axes[0,0].plot(t,res['p_comb'],'r-',label='P_chamber'); axes[0,0].set_ylabel('Pressure (bar)'); axes[0,0].grid(ls='--',lw=0.5); axes[0,0].legend(); axes[0,0].set_title('Pressures')
    axes[1,0].plot(t,res['mdot_ox'],'b-',label='ṁ_ox'); axes[1,0].plot(t,res['mdot_fuel'],'y-',label='ṁ_fuel'); axes[1,0].set_ylabel('Mass Flow (kg/s)'); axes[1,0].grid(ls='--',lw=0.5); axes[1,0].legend(); axes[1,0].set_title('Mass Flow Rates')
    axes[0,1].plot(t,res['of'],'c-'); axes[0,1].set_ylabel('O/F Ratio'); axes[0,1].grid(ls='--',lw=0.5); axes[0,1].set_title('Oxidizer/Fuel Ratio')
    axes[1,1].plot(t,res['r_dot'],'k-'); axes[1,1].set_ylabel('Regression Rate (mm/s)'); axes[1,1].grid(ls='--',lw=0.5); axes[1,1].set_title('Fuel Regression Rate')
    m_tot_ox = np.cumsum(np.array(res['mdot_ox'])*(t[1]-t[0])); m_tot_fuel = np.cumsum(np.array(res['mdot_fuel'])*(t[1]-t[0]))
    axes[2,0].plot(t,m_tot_ox,'b-',label='Oxidizer'); axes[2,0].plot(t,m_tot_fuel,'y-',label='Fuel'); axes[2,0].set_xlabel('Time (s)'); axes[2,0].set_ylabel('Cumulative Mass (kg)'); axes[2,0].grid(ls='--',lw=0.5); axes[2,0].legend(); axes[2,0].set_title('Cumulative Mass Expelled')
    axes[2,1].plot(t, res['thrust'], 'r-'); axes[2,1].set_xlabel('Time (s)'); axes[2,1].set_ylabel('Thrust (N)'); axes[2,1].grid(ls='--',lw=0.5); axes[2,1].set_title('Thrust (Ideal, no nozzle correction)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()

def check_thread_progress():
    try:
        message = result_queue.get_nowait()
        if 'progress' in message:
            progress_bar['value'] = message['progress']; status_var.set(f"Calculating... {message['progress']:.1f}%")
            root.after(100, check_thread_progress)
        elif 'results' in message:
            status_var.set("Done. Generating plots...")
            progress_bar['value'] = 100
            plot_simulation_results(message['results'], params['model_name'])
            summary = "\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in message['results']['summary'].items())
            messagebox.showinfo("Simulation Summary", summary)
            status_var.set("Finished."); reset_ui_after_run()
        elif 'warning' in message:
            messagebox.showwarning("Simulation Warning", message['warning'])
        elif 'error' in message:
            messagebox.showerror("Calculation Error", message['error'])
            status_var.set("Error occurred."); reset_ui_after_run()
    except queue.Empty: root.after(100, check_thread_progress)

def reset_ui_after_run():
    progress_bar.grid_remove()
    btn_run_time.config(state='normal'); btn_run_dp.config(state='normal')

# ---------- E. GUI SETUP AND LOGIC ----------
root = tk.Tk(); root.title("Full Hybrid Rocket System Modeller"); root.geometry("500x820")
main_notebook = ttk.Notebook(root); main_notebook.pack(expand=True, fill='both', padx=5, pady=5)
feed_frame = ttk.Frame(main_notebook); main_notebook.add(feed_frame, text='Feed & General')
input_frame = tk.LabelFrame(feed_frame, text="Model Inputs", padx=10, pady=10); input_frame.pack(fill="x", padx=10, pady=10)
row=0
tk.Label(input_frame,text="Upstream Tank Pressure (P1):").grid(row=row,column=0,sticky="w"); entry_P1=tk.Entry(input_frame); entry_P1.insert(0,"60"); entry_P1.grid(row=row,column=1,sticky="ew")
combo_P1unit=ttk.Combobox(input_frame,values=["psi","bar"],width=5,state='readonly'); combo_P1unit.set("bar"); combo_P1unit.grid(row=row,column=2,padx=5); row+=1
tk.Label(input_frame, text="Feed System Pipe Loss:").grid(row=row, column=0, sticky="w"); entry_PipeLoss = tk.Entry(input_frame); entry_PipeLoss.insert(0, "2.5"); entry_PipeLoss.grid(row=row, column=1, sticky="ew")
combo_PipeLossUnit = ttk.Combobox(input_frame, values=["bar", "psi"], width=5, state='readonly'); combo_PipeLossUnit.set("bar"); combo_PipeLossUnit.grid(row=row, column=2, padx=5); row+=1
temp_label=tk.StringVar(value="Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]"); tk.Label(input_frame,textvariable=temp_label,justify='left').grid(row=row,column=0,sticky="w")
entry_T1=tk.Entry(input_frame); entry_T1.insert(0,"10"); entry_T1.grid(row=row,column=1,sticky="ew")
combo_T1unit=ttk.Combobox(input_frame,values=["C","K"],width=5,state='readonly'); combo_T1unit.set("C"); combo_T1unit.grid(row=row,column=2,padx=5); row+=1
tk.Label(input_frame,text="Injector Diameter (mm):").grid(row=row,column=0,sticky="w"); entry_D=tk.Entry(input_frame); entry_D.insert(0,"1.0"); entry_D.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Discharge Coefficient (Cd):").grid(row=row,column=0,sticky="w"); entry_Cd=tk.Entry(input_frame); entry_Cd.insert(0,"0.66"); entry_Cd.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Number of Ports:").grid(row=row,column=0,sticky="w"); entry_ports=tk.Entry(input_frame); entry_ports.insert(0,"4"); entry_ports.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Tank Volume (L):").grid(row=row,column=0,sticky="w"); entry_tank_vol=tk.Entry(input_frame); entry_tank_vol.insert(0,"4.54"); entry_tank_vol.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
input_frame.columnconfigure(1,weight=1)

hrap_frame = ttk.Frame(main_notebook); main_notebook.add(hrap_frame, text='Combustion (HRAP)')
hrap_input_frame = tk.LabelFrame(hrap_frame, text="Combustion & Motor Geometry", padx=10, pady=10); hrap_input_frame.pack(fill='x', padx=10, pady=10)
hrap_row=0
tk.Label(hrap_input_frame, text="Reg. Coeff. 'a' (for mm/s):").grid(row=hrap_row,column=0,sticky='w'); entry_a=tk.Entry(hrap_input_frame); entry_a.insert(0,"0.155"); entry_a.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Reg. Exp. 'n' (dimensionless):").grid(row=hrap_row,column=0,sticky='w'); entry_n=tk.Entry(hrap_input_frame); entry_n.insert(0,"0.5"); entry_n.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Fuel Density (kg/m³):").grid(row=hrap_row,column=0,sticky='w'); entry_fuel_rho=tk.Entry(hrap_input_frame); entry_fuel_rho.insert(0,"820"); entry_fuel_rho.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Grain Length (m):").grid(row=hrap_row,column=0,sticky='w'); entry_grain_L=tk.Entry(hrap_input_frame); entry_grain_L.insert(0,"0.13"); entry_grain_L.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Initial Port ID (mm):").grid(row=hrap_row, column=0, sticky='w'); entry_port_id=tk.Entry(hrap_input_frame); entry_port_id.insert(0, "23"); entry_port_id.grid(row=hrap_row, column=1, sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Fuel OD (mm):").grid(row=hrap_row, column=0, sticky='w'); entry_fuel_od=tk.Entry(hrap_input_frame); entry_fuel_od.insert(0, "47"); entry_fuel_od.grid(row=hrap_row, column=1, sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Nozzle Throat Diam. (mm):").grid(row=hrap_row,column=0,sticky='w'); entry_nozzle_d=tk.Entry(hrap_input_frame); entry_nozzle_d.insert(0,"11.61"); entry_nozzle_d.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
tk.Label(hrap_input_frame, text="Char. Velocity c* (m/s):").grid(row=hrap_row,column=0,sticky='w'); entry_c_star=tk.Entry(hrap_input_frame); entry_c_star.insert(0,"1580"); entry_c_star.grid(row=hrap_row,column=1,sticky='ew'); hrap_row+=1
hrap_input_frame.columnconfigure(1, weight=1)

sim_frame = ttk.Frame(main_notebook); main_notebook.add(sim_frame, text='Simulation Control')
dp_frame=tk.LabelFrame(sim_frame,text="Injector Characterization Plot",padx=10,pady=10); dp_frame.pack(fill="x",padx=10,pady=5)
tk.Label(dp_frame,text="ΔP Unit:").pack(side='left'); combo_dpunit=ttk.Combobox(dp_frame,values=["psi","bar"],width=8,state='readonly'); combo_dpunit.set("bar"); combo_dpunit.pack(side='left',padx=5)
btn_run_dp=tk.Button(dp_frame,text="Run ΔP Plot",bg="#84c784",command=lambda:on_run_dp()); btn_run_dp.pack(side='right')
time_frame=tk.LabelFrame(sim_frame,text="Full System Simulation",padx=10,pady=10); time_frame.pack(fill="x",padx=10,pady=5)
time_row=0
tk.Label(time_frame,text="Injector Model:").grid(row=time_row,column=0,sticky="w"); model_options=["Dyer", "SPI (Incompressible)","Nino & Razavi","Burnell (Choked)"]  # Add more models as needed
combo_time_model=ttk.Combobox(time_frame,values=model_options,width=20,state='readonly'); combo_time_model.set("Dyer"); combo_time_model.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
tk.Label(time_frame,text="Simulation Duration (s):").grid(row=time_row,column=0,sticky="w"); entry_duration=tk.Entry(time_frame); entry_duration.insert(0,"6"); entry_duration.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
blowdown_var=tk.BooleanVar(value=True); check_blowdown=tk.Checkbutton(time_frame,text="Enable Tank Blowdown",variable=blowdown_var,command=lambda:toggle_blowdown_widgets()); check_blowdown.grid(row=time_row,column=0,columnspan=3,sticky='w'); time_row+=1
label_initial_mass=tk.Label(time_frame,text="Initial N₂O Mass (kg):"); entry_initial_mass=tk.Entry(time_frame); entry_initial_mass.insert(0,"3.5")
label_initial_mass.grid(row=time_row,column=0,sticky="w"); entry_initial_mass.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
btn_run_time=tk.Button(time_frame,text="Run Full System Simulation",bg="#84a9c7",command=lambda:on_run_time()); btn_run_time.grid(row=time_row,column=0,columnspan=3,sticky='ew',pady=5); time_row+=1
status_var=tk.StringVar(value="Ready."); progress_bar=ttk.Progressbar(time_frame,orient='horizontal',length=100,mode='determinate'); progress_bar.grid(row=time_row,column=0,columnspan=3,sticky='ew',pady=5)
status_label=tk.Label(time_frame,textvariable=status_var,anchor='center'); status_label.grid(row=time_row+1,column=0,columnspan=3,sticky='ew')
progress_bar.grid_remove(); status_label.grid_remove(); time_frame.columnconfigure(1,weight=1)

def on_T1unit_change(_):
    if combo_T1unit.get()=="C": temp_label.set("Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"10")
    else: temp_label.set("Upstream Tank Temp (T1):\n[N2O: 182 to 309 K]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"283.15")
combo_T1unit.bind("<<ComboboxSelected>>",on_T1unit_change)

def toggle_blowdown_widgets():
    is_blow = blowdown_var.get()
    entry_P1.config(state='disabled' if is_blow else 'normal', bg='#E0E0E0' if is_blow else 'white')
    if is_blow: label_initial_mass.grid(); entry_initial_mass.grid()
    else: label_initial_mass.grid_remove(); entry_initial_mass.grid_remove()

def validate_temp(T_input,unit):
    T_c = T_input if unit.upper()=="C" else T_input-273.15
    min_T, max_T = CP.PropsSI('Tmin','N2O')-273.15, CP.PropsSI('Tcrit','N2O')-273.15
    if not (min_T <= T_c <= max_T):
        messagebox.showerror("Input Error",f"Temp must be between {min_T:.1f}°C and {max_T:.1f}°C."); return None
    return T_c

def on_run_dp():
    try:
        T1_c = validate_temp(float(entry_T1.get()), combo_T1unit.get())
        if T1_c is None: return
        P1_val = float(entry_P1.get()); P1_bar = P1_val/14.5038 if combo_P1unit.get()=="psi" else P1_val
        pipe_loss_val = float(entry_PipeLoss.get()); pipe_loss_bar = pipe_loss_val/14.5038 if combo_PipeLossUnit.get()=='psi' else pipe_loss_val
        P_injector_inlet_bar = P1_bar - pipe_loss_bar
        if P_injector_inlet_bar<=0: messagebox.showerror("Input Error","Pressure after pipe loss must be positive."); return
        run_models_vs_dp(P_injector_inlet_bar, T1_c, float(entry_D.get()), float(entry_Cd.get()), int(entry_ports.get()), combo_dpunit.get())
    except Exception as e: messagebox.showerror("Input Error", f"Invalid input for ΔP plot.\n{e}")

def on_run_time():
    global params, result_queue
    try:
        T1_c = validate_temp(float(entry_T1.get()), combo_T1unit.get());
        if T1_c is None: return
        
        if float(entry_port_id.get()) >= float(entry_fuel_od.get()):
            messagebox.showerror("Input Error", "Initial Port ID cannot be greater than or equal to Fuel OD.")
            return

        params = {
            'is_blowdown': blowdown_var.get(), 'T_c': T1_c, 'fluid': 'N2O', 'model_name': combo_time_model.get(),
            'D_mm': float(entry_D.get()), 'Cd': float(entry_Cd.get()), 'n_ports': int(entry_ports.get()),
            'duration_s': float(entry_duration.get()),
            'pipe_loss_bar': float(entry_PipeLoss.get())/14.5038 if combo_PipeLossUnit.get()=='psi' else float(entry_PipeLoss.get()),
            'hrap': {
                'a': float(entry_a.get()), 'n': float(entry_n.get()), 'fuel_density_kg_m3': float(entry_fuel_rho.get()),
                'grain_length_m': float(entry_grain_L.get()), 'initial_port_id_mm': float(entry_port_id.get()),
                'fuel_od_mm': float(entry_fuel_od.get()),
                'nozzle_throat_diam_mm': float(entry_nozzle_d.get()), 'c_star_m_s': float(entry_c_star.get())
            }}
        
        if params['is_blowdown']:
            params['V_tank_m3'] = float(entry_tank_vol.get())/1000.0
            params['initial_mass_kg'] = float(entry_initial_mass.get())
        else:
            P1_val = float(entry_P1.get())
            params['P1_bar'] = P1_val/14.5038 if combo_P1unit.get()=="psi" else P1_val
            
        progress_bar['value'] = 0; status_var.set("Initializing...")
        progress_bar.grid(); status_label.grid()
        btn_run_time.config(state='disabled'); btn_run_dp.config(state='disabled')
        result_queue = queue.Queue()
        threading.Thread(target=run_threaded_simulation, args=(params, result_queue), daemon=True).start()
        root.after(100, check_thread_progress)
        
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input for Time plot.\n{traceback.format_exc()}")
        status_var.set("Input Error."); reset_ui_after_run()

toggle_blowdown_widgets()
root.mainloop()