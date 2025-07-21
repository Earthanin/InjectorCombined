# --------------------------------------------------------------
# Hybrid-Injector Modeller + Tank Blow-Down GUI
# (c) 2023-25  •  Based on original script by Aadam Awad + your mods
# Blowdown logic updated based on the CSI model.
# --------------------------------------------------------------
# – Four injector models: SPI • Dyer NHNE • Nino-Razavi • Burnell
# – GUI (Tkinter) for ΔP plots and time-domain simulation
# – NEW: physically-based blow-down solver that conserves mass & energy
# --------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import math, pandas as pd, sys, traceback
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

# ---------- A.  CORE INJECTOR MODEL FUNCTIONS ----------
def get_fluid_properties(fluid, T_c, P_bar):
    """Retrieves key fluid properties from CoolProp."""
    T_k, P_pa = T_c + 273.15, P_bar * 1e5
    try:
        return {
            'rho'       : CP.PropsSI('D', 'T', T_k, 'P', P_pa, fluid),
            'h'         : CP.PropsSI('H', 'T', T_k, 'P', P_pa, fluid),
            's'         : CP.PropsSI('S', 'T', T_k, 'P', P_pa, fluid),
            'P_vap_bar' : CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 1e5
        }
    except ValueError:
        return None

def get_spi_mdot(P1_bar, T1_c, P2_bar, D_mm, n_ports, Cd, fluid):
    """Calculates mass flow rate using the Standard Incompressible Pipe (SPI) model."""
    props1 = get_fluid_properties(fluid, T1_c, P1_bar)
    if not props1: return 0.0
    ΔP_bar = P1_bar - P2_bar
    if ΔP_bar <= 0: return 0.0
    A_tot = n_ports * math.pi * (D_mm/1000)**2 / 4
    return A_tot * Cd * math.sqrt(2*props1['rho']*ΔP_bar*1e5)

def get_dyer_mdot_at_point(P1_bar, T1_c, P2_bar, D_mm, n_ports, Cd, fluid):
    """Calculates mass flow rate using the Dyer Non-Homogeneous Non-Equilibrium (NHNE) model."""
    props1 = get_fluid_properties(fluid, T1_c, P1_bar)
    if not props1: return 0.0
    ρ1, h1, s1, P_sat = props1['rho'], props1['h'], props1['s'], props1['P_vap_bar']
    ΔP_bar = P1_bar - P2_bar
    if ΔP_bar <= 0: return 0.0
    G_spi = math.sqrt(2*ρ1*ΔP_bar*1e5)

    # sub-cooled / saturated branch
    if P2_bar >= P_sat:
        return Cd * G_spi * n_ports*math.pi*(D_mm/1000)**2/4

    # two-phase branch
    try:
        ρ2 = CP.PropsSI('D','P',P2_bar*1e5,'S',s1,fluid)
        h2 = CP.PropsSI('H','P',P2_bar*1e5,'S',s1,fluid)
        G_hem = ρ2*math.sqrt(max(2*(h1-h2),0))
    except ValueError:
        G_hem = 0.0
    κ = math.sqrt((P1_bar-P2_bar)/max(P_sat-P2_bar,1e-12))
    G_dyer = (κ/(1+κ))*G_spi + (1/(1+κ))*G_hem
    return Cd * G_dyer * n_ports*math.pi*(D_mm/1000)**2/4

# --- Ω-model helpers (Nino & Razavi) ---
def get_dp_dt_sat(T_k, fluid):
    """Calculates the derivative of saturation pressure with respect to temperature."""
    return (CP.PropsSI('P','T',T_k+0.01,'Q',0,fluid) -
            CP.PropsSI('P','T',T_k-0.01,'Q',0,fluid)) / 0.02

def calculate_omega_sat(T_k, fluid):
    """Calculates the dimensionless Omega parameter for the Nino-Razavi model."""
    h_lg = CP.PropsSI('H','T',T_k,'Q',1,fluid) - CP.PropsSI('H','T',T_k,'Q',0,fluid)
    c_pl = CP.PropsSI('C','T',T_k,'Q',0,fluid)
    ρ_l  = CP.PropsSI('D','T',T_k,'Q',0,fluid)
    v_l  = 1/ρ_l
    return (c_pl*T_k*v_l / ((h_lg/1000)**2)) * (get_dp_dt_sat(T_k, fluid)/1000)

def get_supercharging_state(P1_bar, T1_c, fluid):
    """Determines if the flow is in a low or high supercharging state."""
    T_k = T1_c + 273.15
    P_sat_bar = CP.PropsSI('P','T',T_k,'Q',0,fluid)/1e5
    ω = calculate_omega_sat(T_k, fluid)
    η_st = (2*ω)/(1+2*ω)
    return ('high' if P_sat_bar < η_st*P1_bar else 'low'), ω, η_st

def get_low_supercharge_mdot(P1_bar, T1_c, D_mm, n_ports, Cd, fluid, ω):
    """Calculates choked mass flow for the low supercharge case."""
    T_k, P1_pa = T1_c+273.15, P1_bar*1e5
    ρ1 = CP.PropsSI('D','P',P1_pa,'T',T_k,fluid)
    P_sat_pa = CP.PropsSI('P','T',T_k,'Q',0,fluid)
    v1 = 1/ρ1; η_sat = P_sat_pa/P1_pa
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

def get_high_supercharge_mdot(P1_bar,T1_c,D_mm,n_ports,Cd,fluid):
    """Calculates choked mass flow for the high supercharge case."""
    T_k, P1_pa = T1_c+273.15, P1_bar*1e5
    P_sat_pa = CP.PropsSI('P','T',T_k,'Q',0,fluid)
    if P1_pa < P_sat_pa: return 0.0, P_sat_pa/1e5
    ρ1 = CP.PropsSI('D','P',P1_pa,'T',T_k,fluid)
    G = math.sqrt(2*ρ1*(P1_pa-P_sat_pa))
    A_tot = n_ports*math.pi*(D_mm/1000)**2/4
    return Cd*G*A_tot, P_sat_pa/1e5

def get_nino_razavi_mdot(P1_bar,T1_c,P2_bar,D_mm,n_ports,Cd,fluid):
    """Calculates mass flow rate using the full Nino & Razavi flashing flow model."""
    state, ω, _ = get_supercharging_state(P1_bar,T1_c,fluid)
    if state=='high':
        mdot_ch, Pcrit = get_high_supercharge_mdot(P1_bar,T1_c,D_mm,n_ports,Cd,fluid)
    else:
        mdot_ch, Pcrit = get_low_supercharge_mdot(P1_bar,T1_c,D_mm,n_ports,Cd,fluid,ω)
    if mdot_ch is None: return 0.0
    return mdot_ch if (Pcrit is None or P2_bar<=Pcrit) \
           else get_dyer_mdot_at_point(P1_bar,T1_c,P2_bar,D_mm,n_ports,Cd,fluid)

# --- Burnell semi-empirical (choked) ---
def burnell_emp_coeff(P_pa): return -1.5267e-8*P_pa + 0.2279
def get_burnell_mdot(P1_bar,T1_c,D_mm,n_ports,Cd,fluid,**_):
    """Calculates mass flow rate using the Burnell choked flow model."""
    T_k, P_pa = T1_c+273.15, P1_bar*1e5
    ρ = CP.PropsSI("D","P",P_pa,"T",T_k,fluid)
    C  = burnell_emp_coeff(P_pa)
    if C<0: return 0.0
    A_tot = n_ports*math.pi*(D_mm/1000)**2/4
    return Cd*A_tot*math.sqrt(2*ρ*P_pa*C)

# ---------- B.  THERMODYNAMIC BLOW-DOWN SOLVER ----------
def get_initial_tank_conditions(m_liq,T_c,V_tank,fluid):
    """Calculates initial liquid and vapor volumes from mass, temp, and tank volume."""
    T_k = T_c+273.15
    try:
        ρ_liq = CP.PropsSI('D','T',T_k,'Q',0,fluid)
        ρ_vap = CP.PropsSI('D','T',T_k,'Q',1,fluid)
        if abs(ρ_liq-ρ_vap)<1e-9:
            messagebox.showerror("Error","Liquid & vapour densities are identical. Check temperature.")
            return None,None
        # V_liq = (m_total - V_tank*ρ_vap) / (ρ_liq - ρ_vap)
        V_liq = (m_liq - V_tank*ρ_vap)/(ρ_liq-ρ_vap)
        if not (0.001<=V_liq<V_tank):
            messagebox.showerror("Input Error",
                                 f"Inconsistent mass/temp for the given tank volume.\n"
                                 f"Calculated V_liq={V_liq:.4f} m³, which is outside the tank bounds [0.001, {V_tank}] m³.")
            return None,None
    except ValueError as e:
        messagebox.showerror("CoolProp Error", f"Could not calculate initial conditions:\n{e}")
        return None, None
    return V_liq, V_tank-V_liq

def simulate_blowdown(m_liq, v_vap, T_k, mdot, dt, fluid="N2O"):
    """
    Performs one mass-energy balanced blow-down step.
    This logic is inspired by the CSI model (A. Awad) and involves three key stages:
    1. Liquid mass is removed from the tank according to the injector model (mdot).
    2. The remaining vapor expands isothermally into the newly available ullage space,
       causing a pressure drop (approximated by Boyle's Law).
    3. The system is now out of equilibrium. Liquid boils to re-establish saturation
       pressure, consuming its own thermal energy and thus dropping the bulk fluid temperature.
    """
    if m_liq <= 0 or mdot <= 0:
        try:
            P_pa = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid)
            return m_liq, v_vap, T_k, P_pa / 1e5
        except ValueError:
            return m_liq, v_vap, T_k, 0.0

    # --- Step 1 & 2: Remove Mass and Expand Vapor ---
    δm = min(mdot * dt, m_liq)
    try:
        ρ_liq = CP.PropsSI('D', 'T', T_k, 'Q', 0, fluid)
        if ρ_liq <= 0: return m_liq, v_vap, T_k, 0.0
        δV = δm / ρ_liq

        P_pa_current = CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid)
        P_pa_after_expansion = P_pa_current * (v_vap / (v_vap + δV))

        m_liq_after_exp = m_liq - δm
        v_vap_after_exp = v_vap + δV
        T_k_after_exp = T_k
    except ValueError as e:
        print(f"Warning: CoolProp error during expansion step: {e}")
        return m_liq, v_vap, T_k, 0.0

    # --- Step 3: Boil Liquid to Restore Saturation Pressure ---
    m_liq_final = m_liq_after_exp
    v_vap_final = v_vap_after_exp
    T_k_final = T_k_after_exp
    P_pa_final = P_pa_after_expansion

    slice_m = 1e-4  # Amount of mass to boil in each iteration of the inner loop
    max_boil_steps = 200 # Safety break to prevent infinite loops
    for _ in range(max_boil_steps):
        try:
            P_sat_target = CP.PropsSI('P', 'T', T_k_final, 'Q', 0, fluid)
            if P_pa_final >= P_sat_target:
                break # Equilibrium has been reached

            if m_liq_final <= slice_m: break # Not enough liquid left to boil

            h_vap = CP.PropsSI('H','T',T_k_final,'Q',1,fluid) - CP.PropsSI('H','T',T_k_final,'Q',0,fluid)
            c_p_liq = CP.PropsSI('C','T',T_k_final,'Q',0,fluid)
            if c_p_liq <= 0 or h_vap <= 0: break # Unphysical state

            # Energy balance: heat for vaporization comes from cooling the remaining liquid
            ΔT = (slice_m * h_vap) / (m_liq_final * c_p_liq)
            T_k_final -= ΔT
            m_liq_final -= slice_m

            # Volume and pressure adjustments due to boiling
            ρ_vap = CP.PropsSI('D', 'T', T_k_final, 'Q', 1, fluid)
            if ρ_vap > 0:
                v_vap_final += slice_m / ρ_vap
            
            # As mass is boiled into the vapor phase, pressure increases.
            # We approximate this using ideal gas relations for the added mass.
            m_vap_old = (v_vap_final - slice_m/ρ_vap) * CP.PropsSI('D','T',T_k_final+ΔT,'Q',1,fluid)
            if m_vap_old > 0:
                 P_pa_final = P_pa_final * (1 + slice_m / m_vap_old)
            else: # If no initial vapor, pressure becomes the new saturation pressure
                 P_pa_final = CP.PropsSI('P', 'T', T_k_final, 'Q', 0, fluid)

        except (ValueError, ZeroDivisionError):
            break # Exit loop on CoolProp or math errors
    else:
        # This 'else' belongs to the 'for' loop, executes if the loop finished without 'break'
        print("Warning: Max boiling steps reached. Results may be inaccurate.")

    # The final pressure should be the saturation pressure at the final temperature
    try:
        final_P_bar = CP.PropsSI('P', 'T', T_k_final, 'Q', 0, fluid) / 1e5
    except ValueError:
        final_P_bar = P_pa_final / 1e5

    return m_liq_final, v_vap_final, T_k_final, final_P_bar

# ---------- C.  ΔP-SWEEP PLOT ----------
def run_models_vs_dp(P1_bar,T1_c,D_mm,Cd,n_ports,dp_unit,fluid="N2O"):
    """Generates plots comparing injector models against experimental data."""
    data = waxman_exp_data_psi if dp_unit=="psi" else \
           [(p/14.5038,m) for p,m in waxman_exp_data_psi]
    ΔP_exp = np.array([d[0] for d in data])
    m_exp  = np.array([d[1] for d in data])
    m_dyer = []; m_nino=[]
    for dp in ΔP_exp:
        ΔP_bar = dp/14.5038 if dp_unit=="psi" else dp
        P2_bar = P1_bar-ΔP_bar
        m_dyer.append(get_dyer_mdot_at_point(P1_bar,T1_c,P2_bar,
                                             D_mm,n_ports,Cd,fluid))
        m_nino.append(get_nino_razavi_mdot(P1_bar,T1_c,P2_bar,
                                          D_mm,n_ports,Cd,fluid))
    m_dyer,m_nino = np.array(m_dyer),np.array(m_nino)
    err_dyer = np.abs((m_dyer-m_exp)/m_exp)*100
    err_nino = np.abs((m_nino-m_exp)/m_exp)*100
    MAPE_dyer,MAPE_nino = err_dyer.mean(),err_nino.mean()

    plt.figure(figsize=(12,7))
    plt.plot(ΔP_exp,m_exp,'ko',label='Experimental (Waxman 2005)')
    plt.plot(ΔP_exp,m_nino,'r-^',label=f'Nino-Razavi (MAPE {MAPE_nino:.1f}%)')
    plt.plot(ΔP_exp,m_dyer,'g-s',label=f'Dyer (MAPE {MAPE_dyer:.1f}%)')
    plt.title('Mass-Flow vs ΔP',fontsize=16); plt.xlabel(f'ΔP ({dp_unit})')
    plt.ylabel('ṁ (kg/s)'); plt.grid(True,ls='--',lw=0.5); plt.legend(); plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(ΔP_exp,err_nino,'r-^',label='Nino-Razavi')
    plt.plot(ΔP_exp,err_dyer,'g-s',label='Dyer')
    plt.axhline(MAPE_nino,color='r',ls='--')
    plt.axhline(MAPE_dyer,color='g',ls='--')
    plt.title('Percent Error'); plt.xlabel(f'ΔP ({dp_unit})')
    plt.ylabel('%'); plt.grid(True,ls='--',lw=0.5); plt.legend(); plt.show()

# ---------- D.  TIME-DOMAIN SIMULATION ----------
def run_time_simulation(is_blowdown, model_func,
                        P1_bar_user, T1_c, P2_bar,
                        D_mm, Cd, n_ports,
                        duration_s, initial_mass_kg=0, fluid="N2O"):
    """Runs a time-domain simulation for either constant pressure or blowdown."""
    time_vals=[]; mdot_vals=[]; total_m_vals=[]
    p1_vals=[]; p2_vals=[]; Δp_vals=[]

    # --- constant-P branch -------------------------------------------------
    if not is_blowdown:
        mdot = model_func(P1_bar_user,T1_c,P2_bar,D_mm,n_ports,Cd,fluid)
        if mdot<=0:
            messagebox.showinfo("Result","Calculated mass flow rate (ṁ) is zero or negative.")
            return None
        time_vals = np.linspace(0,duration_s,500)
        mdot_vals = np.full_like(time_vals, mdot)
        total_m_vals = mdot*time_vals
        p1_vals = np.full_like(time_vals, P1_bar_user)
        p2_vals = np.full_like(time_vals, P2_bar)
        Δp_vals = np.full_like(time_vals, P1_bar_user-P2_bar)
        final_T_c = T1_c

    # --- blow-down branch --------------------------------------------------
    else:
        V_tank = 0.00454      # m³ (Standard 10-lb N2O bottle – adjust as needed)
        dt     = 0.01         # s (simulation time step)
        V_liq,V_vap = get_initial_tank_conditions(initial_mass_kg,
                                                  T1_c,V_tank,fluid)
        if V_liq is None: return None
        m_liq = initial_mass_kg
        T_k   = T1_c+273.15
        # If user P1 is not provided for blowdown, start at saturation pressure for the given temp
        P_bar = P1_bar_user if P1_bar_user>0 else \
                CP.PropsSI('P','T',T_k,'Q',0,fluid)/1e5
        t=0.0
        while t<duration_s and m_liq>0:
            mdot = model_func(P_bar,T_k-273.15,P2_bar,
                              D_mm,n_ports,Cd,fluid)
            if mdot<=0: break
            
            # Call the blowdown step function to get the state for the next time step
            m_liq,V_vap,T_k,P_bar = simulate_blowdown(m_liq,V_vap,T_k,
                                                     mdot,dt,fluid)
            T1_c = T_k-273.15
            time_vals.append(t)
            mdot_vals.append(mdot)
            total_m_vals.append(initial_mass_kg - m_liq)
            p1_vals.append(P_bar)
            p2_vals.append(P2_bar)
            Δp_vals.append(P_bar - P2_bar)
            t += dt
        final_T_c = T1_c
        if not time_vals:
            messagebox.showinfo("Result","Mass flow rate was zero from the start. No simulation run.")
            return None

    plot_time_simulation_results(time_vals,mdot_vals,total_m_vals,
                                 p1_vals,p2_vals,Δp_vals,combo_time_model.get())
    return {
        "Simulation Steps"   : len(time_vals),
        "Initial P1 (bar)"   : p1_vals[0],
        "Final P1 (bar)"     : p1_vals[-1],
        "Initial ṁ (kg/s)"   : mdot_vals[0],
        "Final ṁ (kg/s)"     : mdot_vals[-1],
        "Total Mass Expelled": total_m_vals[-1],
        "Final Temp (°C)"    : final_T_c
    }

def plot_time_simulation_results(t,mdot,m_tot,p1,p2,Δp,model):
    """Generates plots for the time-domain simulation results."""
    if len(t)<2: return
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(10,12),sharex=True)
    T_start = f"{entry_T1.get()} {combo_T1unit.get()}"
    fig.suptitle(f'Time Simulation ({model})  –  T₁₀={T_start}',fontsize=16)

    ax1.plot(t,mdot); ax1.set_ylabel('ṁ (kg/s)'); ax1.set_title('Instantaneous Mass Flow'); ax1.grid(ls='--',lw=0.5)
    ax2.plot(t,m_tot,'r-'); ax2.set_ylabel('∫ṁ dt (kg)'); ax2.set_title('Cumulative Mass Expelled'); ax2.grid(ls='--',lw=0.5)
    ax3.plot(t,p1,'g-',label='P1 (Upstream)'); ax3.plot(t,p2,'m-',label='P2 (Downstream)'); ax3.plot(t,Δp,'k--',label='ΔP')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Pressure (bar)'); ax3.set_title('Pressures'); ax3.grid(ls='--',lw=0.5); ax3.legend()
    plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

# ---------- E.  GUI SETUP AND LOGIC ----------
root = tk.Tk(); root.title("Hybrid Injector Model"); root.geometry("460x660")
input_frame = tk.LabelFrame(root,text="Model Inputs",padx=10,pady=10); input_frame.pack(fill="x",padx=10,pady=10)

row=0
tk.Label(input_frame,text="Upstream Tank Pressure (P1):").grid(row=row,column=0,sticky="w")
entry_P1=tk.Entry(input_frame); entry_P1.insert(0,"704"); entry_P1.grid(row=row,column=1,sticky="ew")
combo_P1unit=ttk.Combobox(input_frame,values=["psi","bar"],width=5,state='readonly'); combo_P1unit.set("psi"); combo_P1unit.grid(row=row,column=2,padx=5); row+=1

temp_label=tk.StringVar(value="Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]")
tk.Label(input_frame,textvariable=temp_label,justify='left').grid(row=row,column=0,sticky="w")
entry_T1=tk.Entry(input_frame); entry_T1.insert(0,"6.85"); entry_T1.grid(row=row,column=1,sticky="ew")
combo_T1unit=ttk.Combobox(input_frame,values=["C","K"],width=5,state='readonly'); combo_T1unit.set("C"); combo_T1unit.grid(row=row,column=2,padx=5); row+=1
def on_T1unit_change(_):
    if combo_T1unit.get()=="C":
        temp_label.set("Upstream Tank Temp (T1):\n[N2O: -90 to 36 °C]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"6.85")
    else:
        temp_label.set("Upstream Tank Temp (T1):\n[N2O: 182 to 309 K]"); entry_T1.delete(0,'end'); entry_T1.insert(0,"280")
combo_T1unit.bind("<<ComboboxSelected>>",on_T1unit_change)

tk.Label(input_frame,text="Injector Diameter (mm):").grid(row=row,column=0,sticky="w"); entry_D=tk.Entry(input_frame); entry_D.insert(0,"1.5"); entry_D.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Discharge Coefficient (Cd):").grid(row=row,column=0,sticky="w"); entry_Cd=tk.Entry(input_frame); entry_Cd.insert(0,"0.77"); entry_Cd.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="Number of Ports:").grid(row=row,column=0,sticky="w"); entry_ports=tk.Entry(input_frame); entry_ports.insert(0,"1"); entry_ports.grid(row=row,column=1,columnspan=2,sticky="ew"); row+=1
tk.Label(input_frame,text="All calculations use N₂O.",font=("Helvetica",8,"italic")).grid(row=row,column=0,columnspan=3,pady=5)
input_frame.columnconfigure(1,weight=1)

# ΔP comparison frame
dp_frame=tk.LabelFrame(root,text="ΔP Comparison Plot",padx=10,pady=10); dp_frame.pack(fill="x",padx=10,pady=5)
tk.Label(dp_frame,text="Experimental ΔP Unit:").pack(side='left'); combo_dpunit=ttk.Combobox(dp_frame,values=["psi","bar"],width=8,state='readonly'); combo_dpunit.set("psi"); combo_dpunit.pack(side='left',padx=5)
btn_run_dp=tk.Button(dp_frame,text="Run ΔP Plot",bg="#84c784",command=lambda:on_run_dp()); btn_run_dp.pack(side='right')

# Time-simulation frame
time_frame=tk.LabelFrame(root,text="Time Simulation",padx=10,pady=10); time_frame.pack(fill="x",padx=10,pady=5)
time_row=0
tk.Label(time_frame,text="Calculation Model:").grid(row=time_row,column=0,sticky="w")
model_options=["Nino & Razavi","Dyer","Burnell (Choked)","SPI (Incompressible)"]
combo_time_model=ttk.Combobox(time_frame,values=model_options,width=20,state='readonly'); combo_time_model.set("Nino & Razavi"); combo_time_model.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1

tk.Label(time_frame,text="Downstream Pressure (P2):").grid(row=time_row,column=0,sticky="w")
entry_P2=tk.Entry(time_frame); entry_P2.insert(0,"300"); entry_P2.grid(row=time_row,column=1,sticky="ew")
combo_P2unit=ttk.Combobox(time_frame,values=["psi","bar"],width=5,state='readonly'); combo_P2unit.set("psi"); combo_P2unit.grid(row=time_row,column=2,padx=5); time_row+=1

tk.Label(time_frame,text="Simulation Duration (s):").grid(row=time_row,column=0,sticky="w"); entry_duration=tk.Entry(time_frame); entry_duration.insert(0,"10"); entry_duration.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1

blowdown_var=tk.BooleanVar(); check_blowdown=tk.Checkbutton(time_frame,text="Enable Blowdown Simulation",variable=blowdown_var,command=lambda:toggle_blowdown_widgets()); check_blowdown.grid(row=time_row,column=0,columnspan=3,sticky='w'); time_row+=1

label_initial_mass=tk.Label(time_frame,text="Initial N₂O Mass (kg):")
entry_initial_mass=tk.Entry(time_frame); entry_initial_mass.insert(0,"3.5")
label_initial_mass.grid(row=time_row,column=0,sticky="w"); entry_initial_mass.grid(row=time_row,column=1,columnspan=2,sticky="ew"); time_row+=1

btn_run_time=tk.Button(time_frame,text="Run Time Plot",bg="#84a9c7",command=lambda:on_run_time()); btn_run_time.grid(row=time_row,column=0,columnspan=3,sticky='ew',pady=5)
time_frame.columnconfigure(1,weight=1)

# ---------- GUI LOGIC AND HANDLERS ----------
def toggle_blowdown_widgets():
    """Shows/hides widgets based on simulation mode."""
    is_blow = blowdown_var.get()
    # Show/hide initial mass field
    if is_blow:
        label_initial_mass.grid(); entry_initial_mass.grid()
    else:
        label_initial_mass.grid_remove(); entry_initial_mass.grid_remove()
    
    # Disable P2 field if Burnell model is selected (as it's a choked model)
    if combo_time_model.get()=="Burnell (Choked)":
        entry_P2.config(state='disabled',bg='#E0E0E0')
        combo_P2unit.config(state='disabled')
    else:
        entry_P2.config(state='normal',bg='white')
        combo_P2unit.config(state='readonly')

combo_time_model.bind("<<ComboboxSelected>>",lambda _:toggle_blowdown_widgets())
toggle_blowdown_widgets() # Initial call to set state

def validate_temp(T_input,unit):
    """Validates that the temperature is within the valid range for N2O in CoolProp."""
    T_c = T_input if unit.upper()=="C" else T_input-273.15
    if not (-90.67<=T_c<=36.41):
        messagebox.showerror("Input Error","Temperature for N₂O must be between −90.7 °C and 36.4 °C.")
        return None
    return T_c

def show_detailed_error(e,inputs):
    """Displays a detailed error message box for debugging."""
    msg = (f"An error occurred during calculation:\n\n"
           f"Error Type: {e.__class__.__name__}\n"
           f"Message: {e}\n\n"
           f"Inputs Used:\n" +
           "\n".join(f"- {k}: {v}" for k,v in inputs.items())+
           "\n\n--- Traceback ---\n"+traceback.format_exc())
    messagebox.showerror("Calculation Error",msg)

def on_run_dp():
    """Callback for the 'Run ΔP Plot' button."""
    inp={}
    try:
        inp['P1']=float(entry_P1.get()); inp['P1_unit']=combo_P1unit.get()
        inp['T1']=float(entry_T1.get()); inp['T1_unit']=combo_T1unit.get()
        inp['D']=float(entry_D.get()); inp['Cd']=float(entry_Cd.get())
        inp['ports']=int(entry_ports.get()); inp['dp_unit']=combo_dpunit.get()
        T1_c = validate_temp(inp['T1'],inp['T1_unit']);   fluid="N2O"
        if T1_c is None: return
        P1_bar = inp['P1']/14.5038 if inp['P1_unit']=="psi" else inp['P1']
        run_models_vs_dp(P1_bar,T1_c,inp['D'],inp['Cd'],inp['ports'],
                         inp['dp_unit'],fluid)
    except Exception as e: show_detailed_error(e,inp)

def on_run_time():
    """Callback for the 'Run Time Plot' button."""
    inp={}; result=None
    model_map={"Nino & Razavi":get_nino_razavi_mdot,
               "Dyer":get_dyer_mdot_at_point,
               "Burnell (Choked)":get_burnell_mdot,
               "SPI (Incompressible)":get_spi_mdot}
    try:
        inp['T1']=float(entry_T1.get()); inp['T1_unit']=combo_T1unit.get()
        inp['D']=float(entry_D.get()); inp['Cd']=float(entry_Cd.get())
        inp['ports']=int(entry_ports.get()); inp['duration']=float(entry_duration.get())
        model_func = model_map[combo_time_model.get()]
        is_blow   = blowdown_var.get()
        T1_c = validate_temp(inp['T1'],inp['T1_unit']);   fluid="N2O"
        if T1_c is None: return

        # P1 handling: For blowdown, P1 can be optional (calculated from T1).
        # For constant pressure, it's required.
        P1_bar = 0.0
        p1_entry=entry_P1.get().strip()
        if is_blow:
            if p1_entry: # Optional override for initial pressure
                P1_val=float(p1_entry); unit=combo_P1unit.get()
                P1_bar = P1_val/14.5038 if unit=="psi" else P1_val
        else: # Required for constant pressure
            if not p1_entry:
                messagebox.showerror("Input Error","P1 is required for constant pressure simulation."); return
            P1_val=float(p1_entry); unit=combo_P1unit.get()
            P1_bar = P1_val/14.5038 if unit=="psi" else P1_val

        # P2 handling
        P2_bar=0.0
        if combo_time_model.get()!="Burnell (Choked)":
            P2_val=float(entry_P2.get()); unit=combo_P2unit.get()
            P2_bar = P2_val/14.5038 if unit=="psi" else P2_val
            if not is_blow and P1_bar<=P2_bar:
                messagebox.showerror("Input Error","P1 must be greater than P2 for non-choked, constant pressure flow."); return

        init_mass=float(entry_initial_mass.get()) if is_blow else 0
        if is_blow and init_mass<=0:
            messagebox.showerror("Input Error","Initial mass must be a positive number for blowdown simulation."); return

        result = run_time_simulation(is_blow,model_func,
                                     P1_bar,T1_c,P2_bar,
                                     inp['D'],inp['Cd'],inp['ports'],
                                     inp['duration'],init_mass)
    except Exception as e: show_detailed_error(e,inp)
    
    if result:
        summary="\n".join(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}"
                          for k,v in result.items())
        print("\n--- SIMULATION SUMMARY ---\n"+summary)
        messagebox.showinfo("Simulation Summary",summary)

root.mainloop()
