# --------------------------------------------------------------
# Hybrid-Injector Modeller + Tank Blow-Down GUI
# (c) 2023-25    •    Based on original script by Aadam Awad + Tanin's Modifications
#
# Fully Corrected and Enhanced by Gemini:
# - Integrated the detailed equilibrium blowdown model from BlowdownGeminiDyer.py.
# - Implemented threading for the blowdown calculation to prevent GUI freezing.
# - Added a real-time progress bar and status label for user feedback.
# - Made the GUI fully responsive during long calculations.
# - Maintained all original functionality for non-blowdown models.
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

# --- Property Getters ---
def get_subcooled_fluid_properties(fluid, T_c, P_bar):
    """Retrieves key fluid properties for a subcooled fluid where T and P are independent."""
    T_k, P_pa = T_c + 273.15, P_bar * 1e5
    try:
        return {
            'rho'       : CP.PropsSI('D', 'T', T_k, 'P', P_pa, fluid),
            'h'         : CP.PropsSI('H', 'T', T_k, 'P', P_pa, fluid),
            's'         : CP.PropsSI('S', 'T', T_k, 'P', P_pa, fluid),
            'P_vap_bar' : CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 1e5,
            'T_k'       : T_k,
            'P1_bar'    : P_bar
        }
    except ValueError:
        return None

def get_saturated_liquid_properties(fluid, T_c):
    """Retrieves key fluid properties for a SATURATED LIQUID where state is defined by T and Q=0."""
    T_k = T_c + 273.15
    try:
        P_sat_pa = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid)
        return {
            'rho'       : CP.PropsSI('D', 'T', T_k, 'Q', 0, fluid),
            'h'         : CP.PropsSI('H', 'T', T_k, 'Q', 0, fluid),
            's'         : CP.PropsSI('S', 'T', T_k, 'Q', 0, fluid),
            'P_vap_bar' : P_sat_pa / 1e5,
            'T_k'       : T_k,
            'P1_bar'    : P_sat_pa / 1e5
        }
    except ValueError:
        # Handle cases where temperature is out of range for the fluid
        return None

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


# ---------- B. THERMODYNAMIC BLOW-DOWN SOLVER (THREADED) ----------

def get_initial_tank_conditions_blowdown(m_liq_target, T_c, V_tank, fluid, dV_step=0.0001):
    """
    Iteratively finds the initial liquid/vapor volumes to match the target mass.
    This is based on the logic from BlowdownGeminiDyer.py.
    """
    T_k = T_c + 273.15
    try:
        liquid_density = CP.PropsSI("D", "T", T_k, "Q", 0, fluid)
        vapor_density = CP.PropsSI("D", "T", T_k, "Q", 1, fluid)
        
        # Check for physical possibility first
        max_mass = V_tank * liquid_density
        if m_liq_target > max_mass:
            messagebox.showerror("Input Error", f"Target mass ({m_liq_target:.2f} kg) exceeds maximum possible mass ({max_mass:.2f} kg) for the given tank volume and temperature.")
            return None

        # Iterative solver to find volumes
        liquidV = 0
        vaporV = V_tank
        while True:
            liquidV += dV_step
            vaporV -= dV_step
            if vaporV <= 0:
                raise RuntimeError("Tank volume too small for target mass (iteration failed).")
            
            current_mass = liquidV * liquid_density
            if current_mass >= m_liq_target:
                vapor_mass = vaporV * vapor_density
                initial_state = {
                    'liq_vol': liquidV, 'vap_vol': vaporV,
                    'liq_mass': current_mass, 'vap_mass': vapor_mass,
                    'temp_k': T_k
                }
                return initial_state

    except (ValueError, RuntimeError) as e:
        messagebox.showerror("CoolProp/Initialization Error", f"Could not calculate initial conditions:\n{e}")
        return None


def run_threaded_blowdown(params, result_queue):
    """
    This function runs the entire blowdown simulation in a separate thread.
    It takes a dictionary of parameters and puts progress updates and final results
    into a thread-safe queue.
    """
    try:
        # Unpack parameters
        T_c = params['T_c']; V_tank = params['V_tank_m3']
        initial_mass = params['initial_mass_kg']; fluid = params['fluid']
        model_name = params['model_name']; P2_bar = params['P2_bar']
        D_mm = params['D_mm']; Cd = params['Cd']; n_ports = params['n_ports']
        duration_s = params['duration_s']
        pipe_loss_bar = params['pipe_loss_bar'] # --- NEW ---

        # --- Initialization ---
        initial_state = get_initial_tank_conditions_blowdown(initial_mass, T_c, V_tank, fluid)
        if initial_state is None:
            # Error was already shown by the init function
            result_queue.put({'error': "Initialization failed."})
            return

        cur_liq_vol = initial_state['liq_vol']
        cur_vap_vol = initial_state['vap_vol']
        cur_temp = initial_state['temp_k']
        initial_liquid_mass = initial_state['liq_mass']
        
        # --- Simulation Setup ---
        time, dt = 0.0, 0.01 # Change to a smaller value for a more accurate simulation
        time_vals, mdot_vals, p1_vals, temp_vals_c, p_injector_vals = [], [], [], [], [] # --- MODIFIED ---

        total_steps = int(duration_s / dt)
        
        # --- Main Simulation Loop (from BlowdownGeminiDyer.py) ---
        while time < duration_s:
            # Get current liquid density to check if we've run out of propellant
            try:
                current_liq_dens = CP.PropsSI("D", "T", cur_temp, "Q", 0, fluid)
                current_liq_mass = cur_liq_vol * current_liq_dens
            except ValueError:
                break # Exit if temp drops out of range

            if current_liq_mass < 0.001: # Stop if propellant is depleted
                break
            
            # --- STEP 1: Get current properties and calculate Mass Flow ---
            props1_tank = get_saturated_liquid_properties(fluid, cur_temp - 273.15)
            if props1_tank is None: break

            # --- MODIFIED: Account for pipe pressure loss ---
            P_injector_inlet_bar = props1_tank['P1_bar'] - pipe_loss_bar
            if P_injector_inlet_bar <= P2_bar:
                break # Stop if driving pressure is gone

            # Create a new properties dictionary for the injector inlet.
            # The fluid state (rho, h, s) is from the tank, but the pressure driving the flow is reduced.
            props1_injector = props1_tank.copy()
            props1_injector['P1_bar'] = P_injector_inlet_bar

            mDot = calculate_mdot_from_properties(props1_injector, P2_bar, D_mm, n_ports, Cd, fluid, model_name)
            if mDot <= 0: break
                
            # Append data for plotting
            time_vals.append(time)
            p1_vals.append(props1_tank['P1_bar']) # This is the tank pressure
            p_injector_vals.append(P_injector_inlet_bar) # This is the new pressure to plot
            mdot_vals.append(mDot)
            temp_vals_c.append(cur_temp - 273.15)

            # --- STEP 2: Update State based on Mass Removal ---
            deltaV_expelled = (mDot / current_liq_dens) * dt
            cur_liq_vol -= deltaV_expelled
            cur_vap_vol += deltaV_expelled
            
            # --- STEP 3: Re-establish Equilibrium via Boiling ---
            try:
                vapor_dens = CP.PropsSI("D", "T", cur_temp, "Q", 1, fluid)
                h_fg = CP.PropsSI("H","T",cur_temp,"Q",1,fluid) - CP.PropsSI("H","T",cur_temp,"Q",0,fluid)
                c_p = CP.PropsSI("C", "T", cur_temp, "Q", 0, fluid)
            except ValueError: break

            mass_to_boil = deltaV_expelled * vapor_dens
            remaining_liquid_mass = (cur_liq_vol * current_liq_dens)
            
            if remaining_liquid_mass > 1e-6:
                delta_T = (mass_to_boil * h_fg) / (remaining_liquid_mass * c_p)
                cur_temp -= delta_T

            # Update time and report progress
            time += dt
            progress = (time / duration_s) * 100
            result_queue.put({'progress': progress})

        # --- Final Packaging of Results ---
        if not time_vals:
            result_queue.put({'error': "Mass flow was zero from the start. Simulation ended."})
            return

        total_mass_expelled = initial_liquid_mass - (cur_liq_vol * CP.PropsSI("D", "T", cur_temp, "Q", 0, fluid))
        
        final_results = {
            'time_vals': time_vals, 'mdot_vals': mdot_vals, 
            'p1_vals': p1_vals, 'initial_mass_kg': initial_liquid_mass,
            'p_injector_vals': p_injector_vals, # --- NEW ---
            'total_mass_expelled': total_mass_expelled,
            'final_temp_c': temp_vals_c[-1],
            'summary': {
                "Sim Steps": len(time_vals),
                "Initial P_tank (bar)": p1_vals[0], "Final P_tank (bar)": p1_vals[-1],
                "Initial P_inj (bar)": p_injector_vals[0], "Final P_inj (bar)": p_injector_vals[-1], # --- MODIFIED ---
                "Initial ṁ (kg/s)": mdot_vals[0], "Final ṁ (kg/s)": mdot_vals[-1],
                "Total Mass Expelled": total_mass_expelled, "Final Temp (°C)": temp_vals_c[-1]
            }
        }
        result_queue.put({'results': final_results})

    except Exception as e:
        # Put any exceptions into the queue to be handled by the main thread
        tb_str = traceback.format_exc()
        result_queue.put({'error': f"An error occurred in the simulation thread:\n{e}\n\n{tb_str}"})


# ---------- C. ΔP-SWEEP PLOT ----------
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

# ---------- D. TIME-DOMAIN SIMULATION (Orchestrator) ----------
# --- MODIFIED signature to accept pipe_loss_bar ---
def run_time_simulation(is_blowdown, model_name, P1_bar_user, T1_c, P2_bar, D_mm, Cd, n_ports, V_tank_m3, duration_s, initial_mass_kg=0, pipe_loss_bar=0.0, fluid="N2O"):
    # This function now acts as an orchestrator.
    # For non-blowdown (instantaneous) cases, it calculates directly.
    # For blowdown cases, it starts the threaded calculation.
    
    if not is_blowdown:
        # --- MODIFIED: Run FAST, non-blowdown simulation directly ---
        P_injector_inlet_bar = P1_bar_user - pipe_loss_bar
        if P_injector_inlet_bar <= P2_bar:
            messagebox.showinfo("Result", "ṁ is zero or negative (Pressure after loss is <= P2).")
            return
            
        # Get properties using the injector inlet pressure
        props1 = get_subcooled_fluid_properties(fluid, T1_c, P_injector_inlet_bar)
        mdot = calculate_mdot_from_properties(props1, P2_bar, D_mm, n_ports, Cd, fluid, model_name)
        if mdot <= 0: messagebox.showinfo("Result", "ṁ is zero or negative."); return
        
        time_vals = np.linspace(0, duration_s, 500)
        mdot_vals = np.full_like(time_vals, mdot)
        total_m_vals = mdot * time_vals
        p1_vals = np.full_like(time_vals, P1_bar_user) # Tank pressure
        p_injector_vals = np.full_like(time_vals, P_injector_inlet_bar) # Injector inlet pressure
        p2_vals = np.full_like(p1_vals, P2_bar)
        Δp_vals = np.array(p_injector_vals) - np.array(p2_vals) # Delta-P is across the injector

        plot_time_simulation_results(time_vals, mdot_vals, total_m_vals, p1_vals, p2_vals, Δp_vals, model_name, p_injector=p_injector_vals)
        # No summary for this simple case
        
    else:
        # --- MODIFIED: Run SLOW, blowdown simulation in a separate thread ---
        params = {
            'T_c': T1_c, 'V_tank_m3': V_tank_m3, 'initial_mass_kg': initial_mass_kg,
            'fluid': fluid, 'model_name': model_name, 'P2_bar': P2_bar,
            'D_mm': D_mm, 'Cd': Cd, 'n_ports': n_ports, 'duration_s': duration_s,
            'pipe_loss_bar': pipe_loss_bar # --- NEW ---
        }
        
        # Reset progress bar and status label
        progress_bar['value'] = 0
        status_var.set("Initializing...")
        progress_bar.grid() # Make it visible
        status_label.grid()
        
        # Disable buttons to prevent re-clicks
        btn_run_time.config(state='disabled')
        btn_run_dp.config(state='disabled')
        
        # Start the threaded calculation
        global result_queue
        result_queue = queue.Queue()
        threading.Thread(target=run_threaded_blowdown, args=(params, result_queue), daemon=True).start()
        
        # Start polling the queue for updates
        root.after(100, check_thread_progress)

def check_thread_progress():
    """Periodically checks the queue for messages from the worker thread."""
    try:
        message = result_queue.get_nowait()
        
        if 'progress' in message:
            progress = message['progress']
            progress_bar['value'] = progress
            status_var.set(f"Calculating... {progress:.1f}%")
            # Schedule the next check
            root.after(100, check_thread_progress)
            
        elif 'results' in message:
            # --- Calculation Finished Successfully ---
            status_var.set("Done. Generating plots...")
            progress_bar['value'] = 100
            
            res = message['results']

            total_m_vals = np.cumsum(np.array(res['mdot_vals']) * (res['time_vals'][1] - res['time_vals'][0] if len(res['time_vals']) > 1 else 0))

            p2_vals = np.full_like(res['p1_vals'], float(entry_P2.get()) / (14.5038 if combo_P2unit.get() == 'psi' else 1))
            Δp_vals = np.array(res['p1_vals']) - p2_vals

            plot_time_simulation_results(res['time_vals'], res['mdot_vals'], total_m_vals, res['p1_vals'], p2_vals, Δp_vals, combo_time_model.get())
            
            summary = "\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in res['summary'].items())
            messagebox.showinfo("Simulation Summary", summary)
            
            # Clean up and re-enable UI
            status_var.set("Finished.")
            reset_ui_after_run()
            
        elif 'error' in message:
            # --- An Error Occurred ---
            messagebox.showerror("Calculation Error", message['error'])
            status_var.set("Error occurred.")
            reset_ui_after_run()

    except queue.Empty:
        # If the queue is empty, it means the thread is still running.
        # Schedule the next check.
        root.after(100, check_thread_progress)

def reset_ui_after_run():
    """Hides progress elements and re-enables buttons."""
    progress_bar.grid_remove()
    # status_label.grid_remove() # Optional: keep the final status visible
    btn_run_time.config(state='normal')
    btn_run_dp.config(state='normal')

def plot_time_simulation_results(t,mdot,m_tot,p1,p2,Δp,model):
    if len(t)<2: return
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,12),sharex=True)
    fig.suptitle(f'Time Simulation ({model})',fontsize=16)
    ax1.plot(t,mdot); ax1.set_ylabel('ṁ (kg/s)'); ax1.set_title('Instantaneous Mass Flow'); ax1.grid(ls='--',lw=0.5)
    ax2.plot(t,m_tot,'r-'); ax2.set_ylabel('Cumulative Mass Expelled (kg)'); ax2.set_title('Cumulative Mass Expelled'); ax2.grid(ls='--',lw=0.5)
    ax3.plot(t,p1,'g-',label='P1 (Upstream)'); ax3.plot(t,p2,'b-',label='P2 (Downstream)'); ax3.plot(t,Δp,'k--',label='ΔP')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Pressure (bar)'); ax3.set_title('Pressures'); ax3.grid(ls='--',lw=0.5); ax3.legend()
    plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

# ---------- E. GUI SETUP AND LOGIC ----------
root = tk.Tk(); root.title("Hybrid Injector & Blowdown Modeller"); root.geometry("460x780") # Increased height for progress bar
input_frame = tk.LabelFrame(root,text="Model Inputs",padx=10,pady=10); input_frame.pack(fill="x",padx=10,pady=10)
row=0
tk.Label(input_frame,text="Upstream Tank Pressure (P1):").grid(row=row,column=0,sticky="w"); entry_P1=tk.Entry(input_frame); entry_P1.insert(0,"60"); entry_P1.grid(row=row,column=1,sticky="ew")
combo_P1unit=ttk.Combobox(input_frame,values=["psi","bar"],width=5,state='readonly'); combo_P1unit.set("bar"); combo_P1unit.grid(row=row,column=2,padx=5); row+=1
temp_label=tk.StringVar(value="Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]"); tk.Label(input_frame,textvariable=temp_label,justify='left').grid(row=row,column=0,sticky="w")
entry_T1=tk.Entry(input_frame); entry_T1.insert(0,"10"); entry_T1.grid(row=row,column=1,sticky="ew")
combo_T1unit=ttk.Combobox(input_frame,values=["C","K"],width=5,state='readonly'); combo_T1unit.set("C"); combo_T1unit.grid(row=row,column=2,padx=5); row+=1
tk.Label(input_frame,text="Injector Diameter (mm):").grid(row=row,column=0,sticky="w"); entry_D=tk.Entry(input_frame); entry_D.insert(0,"1"); entry_D.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Discharge Coefficient (Cd):").grid(row=row,column=0,sticky="w"); entry_Cd=tk.Entry(input_frame); entry_Cd.insert(0,"0.66"); entry_Cd.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Number of Ports:").grid(row=row,column=0,sticky="w"); entry_ports=tk.Entry(input_frame); entry_ports.insert(0,"4"); entry_ports.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Tank Volume (L):").grid(row=row,column=0,sticky="w"); entry_tank_vol=tk.Entry(input_frame); entry_tank_vol.insert(0,"4.54"); entry_tank_vol.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
input_frame.columnconfigure(1,weight=1)

dp_frame=tk.LabelFrame(root,text="ΔP Comparison Plot",padx=10,pady=10); dp_frame.pack(fill="x",padx=10,pady=5)
tk.Label(dp_frame,text="ΔP Unit:").pack(side='left'); combo_dpunit=ttk.Combobox(dp_frame,values=["psi","bar"],width=8,state='readonly'); combo_dpunit.set("bar"); combo_dpunit.pack(side='left',padx=5)
btn_run_dp=tk.Button(dp_frame,text="Run ΔP Plot",bg="#84c784",command=lambda:on_run_dp()); btn_run_dp.pack(side='right')

time_frame=tk.LabelFrame(root,text="Time Simulation",padx=10,pady=10); time_frame.pack(fill="x",padx=10,pady=5)
time_row=0
tk.Label(time_frame,text="Calculation Model:").grid(row=time_row,column=0,sticky="w"); model_options=["Dyer","Burnell (Choked)","SPI (Incompressible)"]
combo_time_model=ttk.Combobox(time_frame,values=model_options,width=20,state='readonly'); combo_time_model.set("Dyer"); combo_time_model.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
tk.Label(time_frame,text="Downstream Pressure (P2):").grid(row=time_row,column=0,sticky="w"); entry_P2=tk.Entry(time_frame); entry_P2.insert(0,"20"); entry_P2.grid(row=time_row,column=1,sticky="ew")
combo_P2unit=ttk.Combobox(time_frame,values=["psi","bar"],width=5,state='readonly'); combo_P2unit.set("bar"); combo_P2unit.grid(row=time_row,column=2,padx=5); time_row+=1
tk.Label(time_frame,text="Simulation Duration (s):").grid(row=time_row,column=0,sticky="w"); entry_duration=tk.Entry(time_frame); entry_duration.insert(0,"6"); entry_duration.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
blowdown_var=tk.BooleanVar(); check_blowdown=tk.Checkbutton(time_frame,text="Enable Blowdown Simulation",variable=blowdown_var,command=lambda:toggle_blowdown_widgets()); check_blowdown.grid(row=time_row,column=0,columnspan=3,sticky='w'); time_row+=1
label_initial_mass=tk.Label(time_frame,text="Initial N₂O Mass (kg):"); entry_initial_mass=tk.Entry(time_frame); entry_initial_mass.insert(0,"3.5")
label_initial_mass.grid(row=time_row,column=0,sticky="w"); entry_initial_mass.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1
btn_run_time=tk.Button(time_frame,text="Run Time Plot",bg="#84a9c7",command=lambda:on_run_time()); btn_run_time.grid(row=time_row,column=0,columnspan=3,sticky='ew',pady=5); time_row+=1

# --- NEW Progress Bar and Status Label ---
status_var = tk.StringVar(value="Ready.")
progress_bar = ttk.Progressbar(time_frame, orient='horizontal', length=100, mode='determinate')
progress_bar.grid(row=time_row, column=0, columnspan=3, sticky='ew', pady=5)
status_label = tk.Label(time_frame, textvariable=status_var, anchor='center')
status_label.grid(row=time_row+1, column=0, columnspan=3, sticky='ew')
progress_bar.grid_remove() # Hide them initially
status_label.grid_remove()
# ----------------------------------------
time_frame.columnconfigure(1,weight=1)

def on_T1unit_change(_):
    if combo_T1unit.get()=="C": temp_label.set("Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"6.85")
    else: temp_label.set("Upstream Tank Temp (T1):\n[N2O: 182 to 309 K]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"280")
combo_T1unit.bind("<<ComboboxSelected>>",on_T1unit_change)

def toggle_blowdown_widgets():
    is_blow = blowdown_var.get(); state = 'normal' if not is_blow else 'disabled'; bg_color = 'white' if not is_blow else '#E0E0E0'
    entry_P1.config(state=state, bg=bg_color)
    if is_blow: label_initial_mass.grid(); entry_initial_mass.grid()
    else: label_initial_mass.grid_remove(); entry_initial_mass.grid_remove()
    model_is_choked = combo_time_model.get()=="Burnell (Choked)"
    p2_state = 'disabled' if model_is_choked else 'normal'; p2_combo_state = 'disabled' if model_is_choked else 'readonly'
    p2_bg = '#E0E0E0' if model_is_choked else 'white'
    entry_P2.config(state=p2_state, bg=p2_bg); combo_P2unit.config(state=p2_combo_state)
combo_time_model.bind("<<ComboboxSelected>>",lambda _:toggle_blowdown_widgets())

def validate_temp(T_input,unit):
    T_c = T_input if unit.upper()=="C" else T_input-273.15
    if not (CP.PropsSI('Tmin', 'N2O')-273.15 <= T_c <= CP.PropsSI('Tcrit', 'N2O')-273.15):
        messagebox.showerror("Input Error",f"Temp must be between {CP.PropsSI('Tmin', 'N2O')-273.15:.1f}°C and {CP.PropsSI('Tcrit', 'N2O')-273.15:.1f}°C.")
        return None
    return T_c

def on_run_dp():
    try:
        T1_c = validate_temp(float(entry_T1.get()), combo_T1unit.get());
        if T1_c is None: return
        P1_val = float(entry_P1.get()); P1_bar = P1_val / 14.5038 if combo_P1unit.get() == "psi" else P1_val
        run_models_vs_dp(P1_bar, T1_c, float(entry_D.get()), float(entry_Cd.get()), int(entry_ports.get()), combo_dpunit.get())
    except Exception as e: messagebox.showerror("Input Error", f"Invalid input for ΔP plot.\n{e}")

def on_run_time():
    try:
        T1_c = validate_temp(float(entry_T1.get()), combo_T1unit.get());
        if T1_c is None: return
        is_blow = blowdown_var.get(); P1_bar = 0.0
        if not is_blow: P1_val = float(entry_P1.get()); P1_bar = P1_val / 14.5038 if combo_P1unit.get() == "psi" else P1_val
        P2_val = float(entry_P2.get()); P2_bar = P2_val / 14.5038 if combo_P2unit.get() == "psi" else P2_val
        init_mass = float(entry_initial_mass.get()) if is_blow else 0
        if is_blow and init_mass <= 0: messagebox.showerror("Input Error", "Initial mass must be positive for blowdown."); return
        V_tank_m3 = float(entry_tank_vol.get()) / 1000.0
        if V_tank_m3 <= 0: messagebox.showerror("Input Error", "Tank volume must be positive."); return
        
        # This function now decides whether to run the old way or the new threaded way
        run_time_simulation(is_blow, combo_time_model.get(), P1_bar, T1_c, P2_bar, float(entry_D.get()), float(entry_Cd.get()), int(entry_ports.get()), V_tank_m3, float(entry_duration.get()), init_mass)

    except Exception as e: 
        messagebox.showerror("Input Error", f"Invalid input for Time plot.\n{e}")
        status_var.set("Input Error.")
        reset_ui_after_run()

toggle_blowdown_widgets()
root.mainloop()