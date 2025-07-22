# ------------------------------------------------------------------------------------
# Blowdown Simulation with Dyer (NHNE) Two-Phase Flow Model
#
# Original blowdown code by Aadam Awad, Columbia Space Initiative Rocketry Team.
# Dyer model implementation adapted from the user-provided MassflowVSTimeGUI.py.
# Integrated and Corrected by Google Gemini.
#
# This program simulates the blowdown of a liquid nitrous oxide tank, modeling
# the decay of tank pressure and mass flow rate over time. It uses the Dyer
# Non-Homogeneous Non-Equilibrium (NHNE) model to calculate the mass flow rate
# through a choked injector, providing a more accurate prediction than simpler models.
# ------------------------------------------------------------------------------------

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
import math
from decimal import Decimal

# -------------------------------------------------------------------------------
# Initial Parameters
# -------------------------------------------------------------------------------
fluids = 'N2O'  # Fluid to be used in the model

# --- TANK AND PROPELLANT ---
targetLiquidMass = 3.5  # Initial liquid nitrous oxide mass (kg)
temperature = 273.15 + 15      # Expected tank temperature at launch (K)
tankV = 0.00454   # Total tank volume (m^3)

# --- INJECTOR PARAMETERS ---
oCount = 4
D_mm = 1              # Injector orifice diameter (mm)
oArea = math.pi*(D_mm/1000)**2/4        # Area of a single orifice (m^2)
Cd = 0.66                # Injector discharge coefficient (dimensionless)

# --- COMBUSTION CHAMBER PRESSURE ---
# The Dyer model requires the downstream pressure to calculate flow rate.
P_chamber_bar = 20 # Convert to bar for calculations

# --- SIMULATION CONTROL ---
timeStep = 0.001         # Time step for the simulation (s)
dV = 0.0001              # Volumetric step for initial condition calculation

# Arrays for the final output
pressArr = []
massFlowArr = []
timeArr = []

# -------------------------------------------------------------------------------
# Fluid Property and Mass Flow Models (Dyer NHNE)
# -------------------------------------------------------------------------------

def get_fluid_properties(fluid, T_k, P_bar):
    """
    Retrieves SATURATED LIQUID properties using CoolProp.
    In a saturated blowdown, the state is defined by Temperature, so we
    explicitly ask for saturated liquid properties (Quality = 0).
    """
    try:
        props = {
            'rho': CP.PropsSI('D', 'T', T_k, 'Q', 0, fluid),
            'h': CP.PropsSI('H', 'T', T_k, 'Q', 0, fluid),
            's': CP.PropsSI('S', 'T', T_k, 'Q', 0, fluid),
            'P_vap_bar': CP.PropsSI('P', 'T', T_k, 'Q', 0, fluid) / 100000
        }
        return props
    except ValueError:
        # Handle cases where temperature is out of range for the fluid
        return None

def get_dyer_mdot_at_point(P1_bar, T1_k, P2_bar, D_mm, num_ports, Cd, fluid):
    """
    Calculates mass flow rate using the Dyer NHNE model for a single point.
    Units: Pressures in bar, Temperature in Kelvin, Diameter in mm.
    """
    props1 = get_fluid_properties(fluid, T1_k, P1_bar)
    if props1 is None: return 0.0
    
    rho1, h1, s1, p_sat_bar = props1['rho'], props1['h'], props1['s'], props1['P_vap_bar']
    
    # The tank pressure P1 should be the saturation pressure.
    delta_p_bar = p_sat_bar - P2_bar
    if delta_p_bar <= 0: return 0.0
    
    G_spi_ideal = math.sqrt(2 * rho1 * (delta_p_bar * 100000))
    
    # Case 1: Subcooled or saturated liquid flow (no flashing in the orifice)
    if P2_bar >= p_sat_bar:
        A_total = num_ports * (math.pi * ((D_mm / 1000.0)**2) / 4.0)
        mdot = Cd * G_spi_ideal * A_total
        return mdot
    
    # Case 2: Two-phase flow (flashing occurs)
    try:
        # Homogeneous Equilibrium Model (HEM) properties
        rho2_hem = CP.PropsSI('D', 'P', P2_bar * 100000, 'S', s1, fluid)
        h2_hem = CP.PropsSI('H', 'P', P2_bar * 100000, 'S', s1, fluid)
        G_hem_ideal = rho2_hem * math.sqrt(2 * (h1 - h2_hem)) if h1 >= h2_hem else 0
    except ValueError:
        G_hem_ideal = 0.0 # Occurs if state is out of bounds
        
    # Weighting factor kappa
    if p_sat_bar <= P2_bar:
        # Avoid division by zero or negative sqrt; treat as single-phase
        kappa = float('inf')
    else:
        kappa = math.sqrt((P1_bar - P2_bar) / (p_sat_bar - P2_bar))

    if math.isinf(kappa):
        G_dyer_ideal = G_spi_ideal
    else:
        G_dyer_ideal = (kappa / (1 + kappa)) * G_spi_ideal + (1 / (1 + kappa)) * G_hem_ideal
        
    A_total = num_ports * (math.pi * ((D_mm / 1000.0)**2) / 4.0)
    mdot_final = Cd * G_dyer_ideal * A_total
    return mdot_final

# -------------------------------------------------------------------------------
# Thermodynamic Helper Functions
# -------------------------------------------------------------------------------

def vaporizationH(temp_k):
    """Enthalpy of vaporization for the fluid at a given temperature."""
    return CP.PropsSI("H", "T", temp_k, "Q", 1, fluids) - CP.PropsSI("H", "T", temp_k, "Q", 0, fluids)

def specificHeat(temp_k):
    """Isobaric specific heat capacity of the liquid."""
    return CP.PropsSI("C", "T", temp_k, "Q", 0, fluids)

# -------------------------------------------------------------------------------
# *Defining Initial States of the Bulk Fluid and Vapor*
# -------------------------------------------------------------------------------
# This function iterates to find the initial liquid and vapor volumes
# that match the target liquid mass at the specified starting temperature.

def initParams(liquidV, vaporV, temp_k, target_mass):
    """Recursively calculates initial tank conditions."""
    newLiquidV = liquidV + dV
    newVaporV = vaporV - dV

    if newVaporV <= 0:
        raise RuntimeError("Tank volume is too small for the target liquid mass.")

    liquid_density = CP.PropsSI("D", "T", temp_k, "Q", 0, fluids)
    vapor_density = CP.PropsSI("D", "T", temp_k, "Q", 1, fluids)
    
    newLiquidMass = newLiquidV * liquid_density
    newVaporMass = newVaporV * vapor_density

    if newLiquidMass >= target_mass:
        return newLiquidV, newVaporV, newLiquidMass, newVaporMass
    
    return initParams(newLiquidV, newVaporV, temp_k, target_mass)

print("Calculating initial conditions...")
liquidVol, vaporVol, liquidMass, vaporMass = initParams(0, tankV, temperature, targetLiquidMass)
print(f"Initial Liquid Volume: {liquidVol:.4f} m^3")
print(f"Initial Vapor Volume (Ullage): {vaporVol:.4f} m^3")
print(f"Initial Liquid Mass: {liquidMass:.2f} kg")
print(f"Initial Total Mass: {liquidMass + vaporMass:.2f} kg")
print("-" * 30)

# -------------------------------------------------------------------------------
# *Blowdown Iterative Model*
# -------------------------------------------------------------------------------
# This is the main simulation loop. It models the system in discrete time steps.
# In each step, it assumes the fluid is in thermodynamic equilibrium, calculates
# the mass expelled, and then calculates the new tank state (temp, pressure)
# resulting from the mass removal and subsequent boiling.

def main_simulation():
    """Performs the blowdown calculations."""
    curLiqVol = liquidVol
    curVapVol = vaporVol
    curTemp = temperature
    curPress = CP.PropsSI("P", "T", temperature, "Q", 0, fluids)
    
    inEquilibrium = True
    time = 0.0
    
    # Calculate injector diameter from area for the Dyer model
    D_m = math.sqrt(4 * oArea / math.pi)
    D_mm = D_m * 1000
    
    print("Starting blowdown simulation...")
    
    # Loop until 99% of the initial liquid mass is expelled
    while curLiqVol * CP.PropsSI("D", "T", curTemp, "Q", 0, fluids) > (liquidMass * 0.01):
        
        if inEquilibrium:
            # --- STEP 1: Calculate Mass Flow for this time step ---
            P1_bar = curPress / 1e5
            T1_k = curTemp
            
            mDot = get_dyer_mdot_at_point(P1_bar, T1_k, P_chamber_bar, D_mm, oCount, Cd, fluids)
            
            if mDot <= 0:
                print("Mass flow is zero or negative. Ending simulation.")
                break
                
            # Append data for plotting
            timeArr.append(time)
            pressArr.append(curPress * 0.000145038) # Convert Pa to PSI for plot
            massFlowArr.append(mDot)
            
            # --- STEP 2: Update State based on Mass Removal ---
            liq_dens = CP.PropsSI("D", "T", curTemp, "Q", 0, fluids)
            deltaV_expelled = (mDot / liq_dens) * timeStep
            
            # Volume of liquid decreases, volume of vapor increases
            curLiqVol -= deltaV_expelled
            curVapVol += deltaV_expelled
            
            # Track time
            time += timeStep
            
            # The removal of mass disturbs the equilibrium
            inEquilibrium = False
            
        else:
            # --- STEP 3: Re-establish Equilibrium via Boiling ---
            # Model the boiling needed to bring the system back to the saturation line.
            # As liquid boils, temperature drops and pressure rises until P_tank = P_sat(T_tank).
            
            # Mass of liquid that needs to boil to fill the new vapor volume
            mass_to_boil = deltaV_expelled * CP.PropsSI("D", "T", curTemp, "Q", 1, fluids)

            # Energy required for this boiling comes from the remaining liquid, reducing its temp
            h_fg = vaporizationH(curTemp)
            c_p = specificHeat(curTemp)
            remaining_liquid_mass = curLiqVol * CP.PropsSI("D", "T", curTemp, "Q", 0, fluids)
            
            if remaining_liquid_mass > 0:
                delta_T = (mass_to_boil * h_fg) / (remaining_liquid_mass * c_p)
                curTemp -= delta_T
            
            # The new pressure is the saturation pressure at the new, lower temperature
            curPress = CP.PropsSI("P", "T", curTemp, "Q", 0, fluids)
            
            inEquilibrium = True

# Run the simulation
main_simulation()
print("Simulation complete.")

# -------------------------------------------------------------------------------
# *Graphical interface*
# -------------------------------------------------------------------------------
if not timeArr:
    print("\nNo data was generated to plot.")
else:
    tankVGal = Decimal(tankV * 264.172)
    
    # --- Plot 1: Pressure vs. Time ---
    plt.figure(1, figsize=(10, 6))
    maxValPress = max(pressArr) if pressArr else 0
    plt.ylim(0, maxValPress + 100)
    plt.plot(timeArr, pressArr, color='b')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Tank Pressure (PSI)', fontsize=12)
    plt.title(f'Tank Pressure vs. Time (Dyer Model)\nInitial Temp: {temperature} K, Chamber P: {P_chamber_bar} Bar', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # --- Plot 2: Mass Flow Rate vs. Time ---
    plt.figure(2, figsize=(10, 6))
    maxValMDot = max(massFlowArr) if massFlowArr else 0
    plt.ylim(0, maxValMDot * 1.1)
    plt.plot(timeArr, massFlowArr, color='r')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Mass Flow Rate (kg/s)', fontsize=12)
    plt.title(f'Mass Flow Rate vs. Time (Dyer Model)\nInitial Temp: {temperature} K, Chamber P: {P_chamber_bar} Bar', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()