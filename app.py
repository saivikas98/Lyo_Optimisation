import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nonsteady Freeze-Drying Model", page_icon="❄️", layout="wide")
st.title("Phase 1: Nonsteady Freeze-Drying Prediction and Optimization")
st.caption(
    "One-dimensional finite-volume moving-interface model based on the uploaded "
    "nonsteady-state and FEM freeze-drying papers."
)

R_GAS = 8.314462618
TORR_TO_PA = 133.322368
CAL_TO_J = 4.184
CM2_TO_M2 = 1e-4
HR_TO_S = 3600.0
G_TO_KG = 1e-3


# ============================================================
# PAPER DEFAULTS / ASSUMPTIONS
# ============================================================
PAPER_DEFAULTS = {
    "rho_frozen": 927.1,                    # kg/m3
    "cp_frozen": 1.94e3,                   # J/kg/K
    "k_frozen": 0.002763 * 1000.0,         # kW/m/K -> W/m/K
    "rho_dry": 55.0,                       # kg/m3
    "cp_dry": 0.2595e3,                    # J/kg/K
    "k_dry_intercept": 2.706e-5 * 1000.0,  # kW/m/K -> W/m/K
    "k_dry_pressure": 8.826e-8 * 1000.0,   # kW/m/K/Pa -> W/m/K/Pa
    "h_top": 0.0042 * 1000.0,              # kW/m2/K -> W/m2/K
    "h_bottom_gap": 0.00873 * 1000.0,       # Process 1, kW/m2/K -> W/m2/K
    "h_bottom_contact": 0.01622 * 1000.0,   # Process 1, kW/m2/K -> W/m2/K
    "h_side_center": 0.0,                  # interior vial
    "h_side_edge": 0.84e-4 * CAL_TO_J * 1e4,  # cal/s/cm2/K -> W/m2/K
    "glass_density": 2230.0,               # engineering assumption, editable
    "glass_cp": 830.0,                     # engineering assumption, editable
    "glass_k": 1.1,                        # engineering assumption, editable
    "glass_thickness": 0.0017,             # m, based on current vial wall input
    "condenser_capacity": 0.0005,           # kg/s per modeled vial equivalent, editable
    "desorption_preexp": 3.34e3,           # 1/s
    "desorption_ea": 8136.0,               # J/mol
    "initial_sorbed_water": 0.2059,         # kg water/kg dry solid
    "Af0": 4.26e-5,
    "Ef1": 1929.0,
    "n_iso": 0.3,
    "AL0": 6.42e-5,
    "BL0": 7.88e-6,
    "Ef2": 5028.0,
    "Ef3": 5028.0,
    "Tg_dry": 348.2,                       # K
    "Tg_water": 135.0,                     # K
    "K_GT": 0.0920,
    "GT_a1": 480.8,
    "GT_a2": 1224.0,
}


# ============================================================
# PHYSICAL FUNCTIONS
# ============================================================
def vapor_pressure_ice_torr(temp_c):
    return np.exp(-6144.96 / (temp_c + 273.15) + 24.01849)


def vapor_pressure_water_pa(temp_k):
    temp_c = temp_k - 273.15
    # Buck equation, adequate for the optional moisture submodel.
    return 611.21 * np.exp((18.678 - temp_c / 234.5) * temp_c / (257.14 + temp_c))


def product_resistance_cm2_hr_torr_g(ldry_cm, r0, a1, a2):
    return r0 + (a1 * ldry_cm) / (1.0 + a2 * ldry_cm)


def kv_cal_s_cm2_k(pc_torr, kc, kp, kd):
    return kc + (kp * pc_torr) / (1.0 + kd * pc_torr)


def kv_to_w_m2_k(kv_value):
    return kv_value * CAL_TO_J * 1e4


def equilibrium_moisture_paper(aw, temp_k, p):
    aw = np.clip(aw, 0.0, 0.999999)
    af = p["Af0"] * np.exp(p["Ef1"] / temp_k)
    al = p["AL0"] * np.exp(p["Ef2"] / temp_k)
    bl = p["BL0"] * np.exp(p["Ef3"] / temp_k)
    value = af * aw ** p["n_iso"] + (al * aw) / (1.0 + bl * aw)
    return float(np.clip(value, 0.0, 1.0))


def glass_transition_k(moisture_kgkg, p):
    # Modified Gordon-Taylor implementation using paper coefficients.
    w = np.clip(moisture_kgkg / (1.0 + moisture_kgkg), 0.0, 0.5)
    dry = 1.0 - w
    base = (dry * p["Tg_dry"] + p["K_GT"] * w * p["Tg_water"]) / (
        dry + p["K_GT"] * w
    )
    return base + p["GT_a1"] * w * dry + p["GT_a2"] * w * w * dry


def arrhenius_desorption_rate(temp_k, p):
    return p["desorption_preexp"] * np.exp(-p["desorption_ea"] / (R_GAS * temp_k))


def recipe_value(recipe, time_hr):
    row = recipe[(recipe["Start (h)"] <= time_hr) & (time_hr < recipe["End (h)"])]
    if row.empty:
        row = recipe.iloc[[-1]]
    row = row.iloc[0]
    return float(row["Shelf T (°C)"]), float(row["Pressure (Torr)"])


def tridiagonal_solve(a, b, c, d):
    n = len(d)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        den = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den
    den = b[-1] - a[-1] * cp[-1]
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / den
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class SimulationResult:
    history: pd.DataFrame
    profiles: dict
    summary: dict


# ============================================================
# NONSTEADY 1D FINITE-VOLUME SOLVER
# ============================================================
def simulate_nonsteady(p, recipe, store_profiles=True):
    n = int(p["nodes"])
    height_m = p["cake_height_m"]
    dz = height_m / (n - 1)
    z = np.linspace(0.0, height_m, n)  # bottom to top
    dt = p["dt_s"]
    area = p["product_area_m2"]
    side_per_volume = 4.0 / p["inner_diameter_m"]

    temp = np.full(n, p["initial_product_temp_c"] + 273.15)
    moisture = np.full(n, p["initial_sorbed_water"])
    dry_layer = 0.0
    elapsed = 0.0
    ice_mass_initial = p["rho_ice"] * area * height_m * (1.0 - p["solid_fraction"])
    ice_mass = ice_mass_initial
    rows = []
    profiles = {}
    endpoint_time = np.nan
    max_temp = -999.0
    max_flux = 0.0
    capacity_exceeded = False
    next_profile = 0.0
    max_time_s = p["max_time_hr"] * 3600.0

    while elapsed <= max_time_s:
        time_hr = elapsed / 3600.0
        shelf_c, pc_torr = recipe_value(recipe, time_hr)
        shelf_k = shelf_c + 273.15
        upper_k = p["upper_surface_temp_c"] + 273.15
        wall_k = p["wall_temp_c"] + 273.15

        dry_mask = z >= (height_m - dry_layer)
        k = np.where(
            dry_mask,
            p["k_dry_intercept"] + p["k_dry_pressure"] * pc_torr * TORR_TO_PA,
            p["k_frozen"],
        )
        rho = np.where(dry_mask, p["rho_dry"], p["rho_frozen"])
        cp = np.where(dry_mask, p["cp_dry"], p["cp_frozen"])

        # Mass transfer at the moving interface, using the conventional Rp basis.
        interface_index = int(np.clip(np.argmin(np.abs(z - (height_m - dry_layer))), 0, n - 1))
        interface_temp_c = temp[interface_index] - 273.15
        pice = vapor_pressure_ice_torr(interface_temp_c)
        rp = product_resistance_cm2_hr_torr_g(
            dry_layer * 100.0, p["r0"], p["a1"], p["a2"]
        )
        mdot_g_hr = max(0.0, p["product_area_cm2"] * (pice - pc_torr) / max(rp, 1e-12))
        mdot_kg_s = mdot_g_hr * G_TO_KG / HR_TO_S
        mdot_kg_s = min(mdot_kg_s, ice_mass / dt if ice_mass > 0 else 0.0)
        mass_flux = mdot_kg_s / area
        latent_flux = mass_flux * p["delta_h_sub_j_kg"]
        max_flux = max(max_flux, mdot_kg_s)
        capacity_exceeded = capacity_exceeded or mdot_kg_s > p["condenser_capacity"]

        # Bottom coefficient includes paper contact/non-contact spatial distribution.
        h_bottom_avg = (
            p["contact_fraction"] * p["h_bottom_contact"]
            + (1.0 - p["contact_fraction"]) * p["h_bottom_gap"]
        )
        # Optional glass-wall resistance in series.
        h_bottom = 1.0 / (
            1.0 / max(h_bottom_avg, 1e-12)
            + p["glass_thickness"] / max(p["glass_k"], 1e-12)
        )
        h_top = p["h_top"]
        h_side = p["h_side_edge"] if p["vial_position"] == "Edge" else p["h_side_center"]

        # Fully implicit finite-volume conduction solve.
        lower = np.zeros(n - 1)
        diag = np.zeros(n)
        upper = np.zeros(n - 1)
        rhs = rho * cp * temp / dt

        for i in range(n):
            storage = rho[i] * cp[i] / dt
            source_coeff = h_side * side_per_volume
            diag[i] = storage + source_coeff
            rhs[i] += source_coeff * wall_k

            if i > 0:
                kw = 2 * k[i] * k[i - 1] / max(k[i] + k[i - 1], 1e-15)
                aw = kw / dz**2
                diag[i] += aw
                lower[i - 1] = -aw
            else:
                ab = h_bottom / dz
                diag[i] += ab
                rhs[i] += ab * shelf_k

            if i < n - 1:
                ke = 2 * k[i] * k[i + 1] / max(k[i] + k[i + 1], 1e-15)
                ae = ke / dz**2
                diag[i] += ae
                upper[i] = -ae
            else:
                at = h_top / dz
                diag[i] += at
                rhs[i] += at * upper_k

        # Sublimation latent heat sink localized at the moving front.
        rhs[interface_index] -= latent_flux / dz
        temp_new = tridiagonal_solve(lower, diag, upper, rhs)

        # Bound-water desorption begins locally after the element becomes dry.
        aw = np.clip(pc_torr * TORR_TO_PA / np.maximum(vapor_pressure_water_pa(temp_new), 1e-12), 0.0, 0.999)
        for i in np.where(dry_mask)[0]:
            cstar = equilibrium_moisture_paper(aw[i], temp_new[i], p)
            kg = arrhenius_desorption_rate(temp_new[i], p)
            moisture[i] = cstar + (moisture[i] - cstar) * np.exp(-kg * dt)

        # Moving interface update.
        ice_removed = mdot_kg_s * dt
        ice_mass = max(0.0, ice_mass - ice_removed)
        dry_layer = min(
            height_m,
            dry_layer + ice_removed / max(p["rho_ice"] * area * (1.0 - p["solid_fraction"]), 1e-15),
        )
        temp = temp_new
        elapsed += dt

        temp_c = temp - 273.15
        tg_k = np.array([glass_transition_k(c, p) for c in moisture])
        safety = tg_k - temp
        max_temp = max(max_temp, float(np.max(temp_c)))
        drying_fraction = 1.0 - ice_mass / max(ice_mass_initial, 1e-15)

        rows.append({
            "Time (h)": elapsed / 3600.0,
            "Shelf T (°C)": shelf_c,
            "Pressure (Torr)": pc_torr,
            "Bottom T (°C)": temp_c[0],
            "Core T (°C)": temp_c[n // 2],
            "Top T (°C)": temp_c[-1],
            "Maximum product T (°C)": np.max(temp_c),
            "Minimum product T (°C)": np.min(temp_c),
            "Interface T (°C)": temp_c[interface_index],
            "Dry layer (cm)": dry_layer * 100.0,
            "Drying (%)": 100.0 * drying_fraction,
            "Sublimation rate (g/h)": mdot_g_hr,
            "Residual moisture mean (kg/kg)": np.mean(moisture),
            "Residual moisture max (kg/kg)": np.max(moisture),
            "Minimum Tg-T margin (°C)": np.min(safety),
            "Condenser load (kg/s)": mdot_kg_s,
        })

        if store_profiles and elapsed >= next_profile:
            profiles[round(elapsed / 3600.0, 3)] = pd.DataFrame({
                "Height (cm)": z * 100.0,
                "Temperature (°C)": temp_c,
                "Moisture (kg/kg)": moisture,
                "Tg-T margin (°C)": safety,
                "Region": np.where(dry_mask, "Dry", "Frozen"),
            })
            next_profile += p["profile_interval_hr"] * 3600.0

        if ice_mass <= 1e-12 and np.isnan(endpoint_time):
            endpoint_time = elapsed / 3600.0

        if ice_mass <= 1e-12 and elapsed / 3600.0 >= recipe["End (h)"].max():
            break

    history = pd.DataFrame(rows)
    summary = {
        "Primary drying endpoint (h)": endpoint_time,
        "Maximum product temperature (°C)": max_temp,
        "Final drying (%)": history["Drying (%)"].iloc[-1],
        "Final mean moisture (kg/kg)": history["Residual moisture mean (kg/kg)"].iloc[-1],
        "Minimum Tg-T margin (°C)": history["Minimum Tg-T margin (°C)"].min(),
        "Maximum vial mass flow (kg/s)": max_flux,
        "Condenser capacity exceeded": capacity_exceeded,
    }
    return SimulationResult(history, profiles, summary)


# ============================================================
# STREAMLIT INPUTS
# ============================================================
with st.sidebar:
    st.header("Geometry and formulation")
    fill_ml = st.number_input("Fill volume (mL)", value=48.0, min_value=0.01)
    d_out_cm = st.number_input("Outer vial diameter (cm)", value=4.7, min_value=0.01)
    wall_cm = st.number_input("Vial wall thickness (cm)", value=0.17, min_value=0.0)
    solid_fraction = st.number_input("Solid fraction", value=0.0625, min_value=0.0, max_value=0.99)
    rho_ice = st.number_input("Ice density (kg/m³)", value=918.0)
    delta_h = st.number_input("Heat of sublimation (kJ/kg)", value=2834.6)
    critical_temp = st.number_input("Critical product temperature (°C)", value=-10.0)
    vial_position = st.selectbox("Vial position", ["Center", "Edge"])

    st.header("Product resistance")
    r0 = st.number_input("R0 (cm²·h·Torr/g)", value=44.59)
    a1 = st.number_input("A1", value=1451.73)
    a2 = st.number_input("A2", value=12.93)

    with st.expander("Paper material and boundary defaults"):
        rho_frozen = st.number_input("Frozen density (kg/m³)", value=PAPER_DEFAULTS["rho_frozen"])
        cp_frozen = st.number_input("Frozen Cp (J/kg/K)", value=PAPER_DEFAULTS["cp_frozen"])
        k_frozen = st.number_input("Frozen k (W/m/K)", value=PAPER_DEFAULTS["k_frozen"], format="%.6f")
        rho_dry = st.number_input("Dry density (kg/m³)", value=PAPER_DEFAULTS["rho_dry"])
        cp_dry = st.number_input("Dry Cp (J/kg/K)", value=PAPER_DEFAULTS["cp_dry"])
        k_dry_intercept = st.number_input("Dry k intercept (W/m/K)", value=PAPER_DEFAULTS["k_dry_intercept"], format="%.6f")
        k_dry_pressure = st.number_input("Dry k pressure coefficient", value=PAPER_DEFAULTS["k_dry_pressure"], format="%.9f")
        h_top = st.number_input("Top radiation h (W/m²/K)", value=PAPER_DEFAULTS["h_top"], format="%.4f")
        h_bottom_gap = st.number_input("Bottom gap h (W/m²/K)", value=PAPER_DEFAULTS["h_bottom_gap"], format="%.4f")
        h_bottom_contact = st.number_input("Bottom contact h (W/m²/K)", value=PAPER_DEFAULTS["h_bottom_contact"], format="%.4f")
        contact_fraction = st.number_input("Bottom contact area fraction", value=0.278, min_value=0.0, max_value=1.0)
        h_side_center = st.number_input("Center side h (W/m²/K)", value=PAPER_DEFAULTS["h_side_center"])
        h_side_edge = st.number_input("Edge side h (W/m²/K)", value=PAPER_DEFAULTS["h_side_edge"], format="%.4f")
        glass_density = st.number_input("Glass density (kg/m³)", value=PAPER_DEFAULTS["glass_density"])
        glass_cp = st.number_input("Glass Cp (J/kg/K)", value=PAPER_DEFAULTS["glass_cp"])
        glass_k = st.number_input("Glass k (W/m/K)", value=PAPER_DEFAULTS["glass_k"])
        condenser_capacity = st.number_input("Condenser capacity per vial equivalent (kg/s)", value=PAPER_DEFAULTS["condenser_capacity"], format="%.7f")

    with st.expander("Secondary drying, sorption, and Tg defaults"):
        initial_sorbed_water = st.number_input("Initial sorbed water (kg/kg)", value=PAPER_DEFAULTS["initial_sorbed_water"])
        desorption_preexp = st.number_input("Desorption pre-exponential (1/s)", value=PAPER_DEFAULTS["desorption_preexp"])
        desorption_ea = st.number_input("Desorption activation energy (J/mol)", value=PAPER_DEFAULTS["desorption_ea"])
        Af0 = st.number_input("Sorption Af0", value=PAPER_DEFAULTS["Af0"], format="%.8f")
        Ef1 = st.number_input("Sorption Ef1", value=PAPER_DEFAULTS["Ef1"])
        n_iso = st.number_input("Sorption n", value=PAPER_DEFAULTS["n_iso"])
        AL0 = st.number_input("Sorption AL0", value=PAPER_DEFAULTS["AL0"], format="%.8f")
        BL0 = st.number_input("Sorption BL0", value=PAPER_DEFAULTS["BL0"], format="%.8f")
        Ef2 = st.number_input("Sorption Ef2", value=PAPER_DEFAULTS["Ef2"])
        Ef3 = st.number_input("Sorption Ef3", value=PAPER_DEFAULTS["Ef3"])
        Tg_dry = st.number_input("Dry solid Tg (K)", value=PAPER_DEFAULTS["Tg_dry"])
        Tg_water = st.number_input("Water Tg (K)", value=PAPER_DEFAULTS["Tg_water"])
        K_GT = st.number_input("Gordon-Taylor K", value=PAPER_DEFAULTS["K_GT"])
        GT_a1 = st.number_input("Modified GT a1", value=PAPER_DEFAULTS["GT_a1"])
        GT_a2 = st.number_input("Modified GT a2", value=PAPER_DEFAULTS["GT_a2"])

    with st.expander("Numerical settings"):
        nodes = st.number_input("Axial nodes", value=31, min_value=11, max_value=101, step=2)
        dt_s = st.number_input("Time step (s)", value=30.0, min_value=1.0)
        initial_product_temp_c = st.number_input("Initial product temperature (°C)", value=-40.0)
        upper_surface_temp_c = st.number_input("Upper shelf/surface temperature (°C)", value=-20.0)
        wall_temp_c = st.number_input("Chamber wall temperature (°C)", value=20.0)
        max_time_hr = st.number_input("Maximum simulated time (h)", value=100.0)
        profile_interval_hr = st.number_input("Profile storage interval (h)", value=4.0, min_value=0.1)

    run = st.button("Run nonsteady simulation", type="primary", use_container_width=True)

inner_cm = d_out_cm - 2.0 * wall_cm
if inner_cm <= 0:
    st.error("Outer diameter must be greater than twice the wall thickness.")
    st.stop()
area_cm2 = np.pi * inner_cm**2 / 4.0
area_m2 = area_cm2 * CM2_TO_M2
cake_height_m = (fill_ml / area_cm2) / 100.0

st.subheader("Process recipe and deviation editor")
default_recipe = pd.DataFrame({
    "Start (h)": [0.0, 4.0, 5.0, 7.0, 12.0],
    "End (h)": [4.0, 5.0, 7.0, 12.0, 100.0],
    "Shelf T (°C)": [-8.0, -3.0, -3.0, -11.0, -8.0],
    "Pressure (Torr)": [0.100, 0.100, 0.150, 0.050, 0.100],
})
recipe = st.data_editor(default_recipe, use_container_width=True, num_rows="dynamic")
recipe = recipe.sort_values("Start (h)").reset_index(drop=True)

p = {
    "product_area_cm2": area_cm2,
    "product_area_m2": area_m2,
    "inner_diameter_m": inner_cm / 100.0,
    "cake_height_m": cake_height_m,
    "solid_fraction": solid_fraction,
    "rho_ice": rho_ice,
    "delta_h_sub_j_kg": delta_h * 1000.0,
    "critical_temp_c": critical_temp,
    "vial_position": vial_position,
    "r0": r0, "a1": a1, "a2": a2,
    "rho_frozen": rho_frozen, "cp_frozen": cp_frozen, "k_frozen": k_frozen,
    "rho_dry": rho_dry, "cp_dry": cp_dry,
    "k_dry_intercept": k_dry_intercept, "k_dry_pressure": k_dry_pressure,
    "h_top": h_top, "h_bottom_gap": h_bottom_gap,
    "h_bottom_contact": h_bottom_contact, "contact_fraction": contact_fraction,
    "h_side_center": h_side_center, "h_side_edge": h_side_edge,
    "glass_density": glass_density, "glass_cp": glass_cp, "glass_k": glass_k,
    "glass_thickness": wall_cm / 100.0,
    "condenser_capacity": condenser_capacity,
    "initial_sorbed_water": initial_sorbed_water,
    "desorption_preexp": desorption_preexp, "desorption_ea": desorption_ea,
    "Af0": Af0, "Ef1": Ef1, "n_iso": n_iso,
    "AL0": AL0, "BL0": BL0, "Ef2": Ef2, "Ef3": Ef3,
    "Tg_dry": Tg_dry, "Tg_water": Tg_water, "K_GT": K_GT,
    "GT_a1": GT_a1, "GT_a2": GT_a2,
    "nodes": int(nodes), "dt_s": dt_s,
    "initial_product_temp_c": initial_product_temp_c,
    "upper_surface_temp_c": upper_surface_temp_c,
    "wall_temp_c": wall_temp_c,
    "max_time_hr": max_time_hr,
    "profile_interval_hr": profile_interval_hr,
}

metrics = st.columns(4)
metrics[0].metric("Product area", f"{area_cm2:.3f} cm²")
metrics[1].metric("Cake height", f"{cake_height_m*100:.3f} cm")
metrics[2].metric("Axial nodes", int(nodes))
metrics[3].metric("Vial position", vial_position)

if run:
    try:
        with st.spinner("Solving nonsteady heat, moving interface, moisture, and Tg histories..."):
            st.session_state.phase1 = simulate_nonsteady(p, recipe)
            st.session_state.phase1_inputs = p.copy()
            st.session_state.phase1_recipe = recipe.copy()
    except Exception as exc:
        st.exception(exc)

if "phase1" not in st.session_state:
    st.info("Edit the recipe if required, then select **Run nonsteady simulation**.")
    st.stop()

result = st.session_state.phase1
history = result.history
summary = result.summary
profiles = result.profiles
used = st.session_state.phase1_inputs
used_recipe = st.session_state.phase1_recipe

st.subheader("Prediction summary")
summary_cols = st.columns(6)
summary_cols[0].metric("Primary drying endpoint", "Not reached" if np.isnan(summary["Primary drying endpoint (h)"]) else f"{summary['Primary drying endpoint (h)']:.2f} h")
summary_cols[1].metric("Maximum product T", f"{summary['Maximum product temperature (°C)']:.2f} °C")
summary_cols[2].metric("Final drying", f"{summary['Final drying (%)']:.2f}%")
summary_cols[3].metric("Final mean moisture", f"{summary['Final mean moisture (kg/kg)']:.4f}")
summary_cols[4].metric("Minimum Tg-T", f"{summary['Minimum Tg-T margin (°C)']:.2f} °C")
summary_cols[5].metric("Capacity check", "Exceeded" if summary["Condenser capacity exceeded"] else "Within limit")

if summary["Maximum product temperature (°C)"] > used["critical_temp_c"]:
    st.warning("The calculated maximum product temperature exceeds the entered critical temperature.")
else:
    st.success("The calculated maximum product temperature remains below the entered critical temperature.")

(
    tab_dashboard, tab_depth, tab_moisture, tab_recipe,
    tab_design, tab_assumptions, tab_data
) = st.tabs([
    "Combined dashboard", "Depth profiles", "Moisture and Tg",
    "Recipe/deviation", "Ts-Pc design space", "Assumptions and units", "Data"
])

with tab_dashboard:
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    t = history["Time (h)"]
    ax[0,0].plot(t, history["Shelf T (°C)"], "--", label="Shelf")
    ax[0,0].plot(t, history["Bottom T (°C)"], label="Bottom")
    ax[0,0].plot(t, history["Core T (°C)"], label="Core")
    ax[0,0].plot(t, history["Top T (°C)"], label="Top")
    ax[0,0].axhline(used["critical_temp_c"], color="r", ls="--", label="Critical T")
    ax[0,0].set_title("Nonsteady Product Temperature")
    ax[0,0].set_ylabel("Temperature (°C)")
    ax[0,0].legend()
    ax[0,1].plot(t, history["Dry layer (cm)"])
    ax[0,1].set_title("Moving Sublimation Interface")
    ax[0,1].set_ylabel("Dry-layer thickness (cm)")
    ax[1,0].plot(t, history["Drying (%)"])
    ax[1,0].set_title("Primary Drying Progress")
    ax[1,0].set_ylabel("Drying (%)")
    ax[1,1].plot(t, history["Sublimation rate (g/h)"])
    ax[1,1].set_title("Sublimation Rate")
    ax[1,1].set_ylabel("g/h")
    for a in ax.flat:
        a.set_xlabel("Time (h)")
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_depth:
    available = list(profiles.keys())
    chosen = st.multiselect("Stored profile times (h)", available, default=available[-min(4, len(available)):])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for time_key in chosen:
        profile = profiles[time_key]
        axes[0].plot(profile["Temperature (°C)"], profile["Height (cm)"], label=f"{time_key:g} h")
        axes[1].plot(profile["Moisture (kg/kg)"], profile["Height (cm)"], label=f"{time_key:g} h")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Height from vial bottom (cm)")
    axes[0].set_title("Axial Temperature Distribution")
    axes[1].set_xlabel("Moisture (kg/kg dry solid)")
    axes[1].set_ylabel("Height from vial bottom (cm)")
    axes[1].set_title("Axial Moisture Distribution")
    for a in axes:
        a.grid(True, alpha=0.3)
        a.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_moisture:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["Time (h)"], history["Residual moisture mean (kg/kg)"], label="Mean")
    axes[0].plot(history["Time (h)"], history["Residual moisture max (kg/kg)"], "--", label="Maximum")
    axes[0].set_title("Residual Moisture History")
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("kg water/kg dry solid")
    axes[0].legend()
    axes[1].plot(history["Time (h)"], history["Minimum Tg-T margin (°C)"])
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_title("Minimum Glass-Transition Safety Margin")
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Tg - T (°C)")
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_recipe:
    st.dataframe(used_recipe, use_container_width=True, hide_index=True)
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.step(history["Time (h)"], history["Shelf T (°C)"], where="post", color="darkred", label="Shelf T")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Shelf temperature (°C)", color="darkred")
    ax2 = ax1.twinx()
    ax2.step(history["Time (h)"], history["Pressure (Torr)"], where="post", color="navy", label="Pressure")
    ax2.set_ylabel("Pressure (Torr)", color="navy")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with tab_design:
    st.caption("Runs a reduced grid using the same nonsteady model and constant primary-drying conditions.")
    c1, c2, c3 = st.columns(3)
    ts_values = c1.multiselect("Shelf temperatures (°C)", [-20, -15, -13, -11, -8, -5, -3], default=[-13, -8, -3])
    pc_values = c2.multiselect("Pressures (Torr)", [0.05, 0.07, 0.10, 0.13, 0.15], default=[0.07, 0.10, 0.13])
    max_allowed_time = c3.number_input("Maximum acceptable drying time (h)", value=75.0)
    if st.button("Calculate design space"):
        design_rows = []
        progress = st.progress(0)
        total = max(1, len(ts_values) * len(pc_values))
        count = 0
        design_p = used.copy()
        design_p["nodes"] = min(int(used["nodes"]), 21)
        design_p["dt_s"] = max(used["dt_s"], 60.0)
        design_p["profile_interval_hr"] = 1e9
        for ts_const in ts_values:
            for pc_const in pc_values:
                rcp = pd.DataFrame({"Start (h)":[0.0], "End (h)":[used["max_time_hr"]], "Shelf T (°C)":[ts_const], "Pressure (Torr)":[pc_const]})
                sim = simulate_nonsteady(design_p, rcp, store_profiles=False)
                endpoint = sim.summary["Primary drying endpoint (h)"]
                tmax = sim.summary["Maximum product temperature (°C)"]
                feasible = (
                    not np.isnan(endpoint)
                    and endpoint <= max_allowed_time
                    and tmax <= used["critical_temp_c"]
                    and not sim.summary["Condenser capacity exceeded"]
                )
                design_rows.append({"Shelf T (°C)":ts_const, "Pressure (Torr)":pc_const, "Drying time (h)":endpoint, "Maximum product T (°C)":tmax, "Feasible":feasible})
                count += 1
                progress.progress(count / total)
        st.session_state.design_df = pd.DataFrame(design_rows)
    if "design_df" in st.session_state:
        design = st.session_state.design_df
        st.dataframe(design, use_container_width=True, hide_index=True)
        feasible = design[design["Feasible"]]
        if not feasible.empty:
            best = feasible.sort_values("Drying time (h)").iloc[0]
            st.success(f"Fastest feasible point: Ts = {best['Shelf T (°C)']:.1f} °C, Pc = {best['Pressure (Torr)']:.3f} Torr, time = {best['Drying time (h)']:.2f} h")
        else:
            st.warning("No tested combination met all constraints.")

with tab_assumptions:
    st.subheader("Reference-paper defaults and engineering assumptions")
    assumptions = pd.DataFrame([
        ["Frozen density", used["rho_frozen"], "kg/m³", "Paper Table 2"],
        ["Frozen Cp", used["cp_frozen"], "J/kg/K", "Paper Table 2"],
        ["Frozen k", used["k_frozen"], "W/m/K", "Paper Table 2, converted from kW/m/K"],
        ["Dry density", used["rho_dry"], "kg/m³", "Paper Table 2"],
        ["Dry Cp", used["cp_dry"], "J/kg/K", "Paper Table 2"],
        ["Dry k", "pressure dependent", "W/m/K", "Paper Table 2"],
        ["Top radiation h", used["h_top"], "W/m²/K", "Paper Table 2"],
        ["Bottom gap h", used["h_bottom_gap"], "W/m²/K", "Paper Process 1"],
        ["Bottom contact h", used["h_bottom_contact"], "W/m²/K", "Paper Process 1"],
        ["Side radiation", used["h_side_edge"], "W/m²/K", "Derived from paper radiation term"],
        ["Glass properties", "editable", "SI", "Engineering assumptions; paper notes glass heat capacity was omitted"],
        ["Condenser capacity", used["condenser_capacity"], "kg/s/vial-equivalent", "Editable engineering assumption"],
        ["Desorption kinetics", f"A={used['desorption_preexp']}, Ea={used['desorption_ea']}", "1/s, J/mol", "Paper Figure 5"],
        ["Sorption isotherm", "paper coefficients", "kg/kg", "Paper Table 1"],
        ["Glass transition", "modified Gordon-Taylor", "K", "Paper Table 1"],
    ], columns=["Item", "Value", "Unit", "Basis"])
    st.dataframe(assumptions, use_container_width=True, hide_index=True)
    st.warning("This Phase 1 solver is a one-dimensional finite-volume implementation, not the paper's full two-dimensional axisymmetric ALE finite-element solver. Side radiation is represented as a distributed source, and the interface is planar.")

with tab_data:
    st.dataframe(history, use_container_width=True, height=500)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        history.to_excel(writer, sheet_name="History", index=False)
        used_recipe.to_excel(writer, sheet_name="Recipe", index=False)
        pd.DataFrame(summary.items(), columns=["Metric", "Value"]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(used.items(), columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Inputs", index=False)
        for i, (key, profile) in enumerate(profiles.items()):
            profile.to_excel(writer, sheet_name=f"Profile_{i+1}", index=False)
    c1, c2 = st.columns(2)
    c1.download_button("Download history CSV", history.to_csv(index=False).encode("utf-8"), "phase1_nonsteady_history.csv", "text/csv", use_container_width=True)
    c2.download_button("Download complete Excel", output.getvalue(), "phase1_nonsteady_freeze_drying.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.divider()
st.caption("Research model. Validate against formulation-specific Rp, Kv, thermocouple/MTM data, primary-drying endpoint, and moisture measurements before process decisions.")
