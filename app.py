import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

st.set_page_config(
    page_title="Integrated Freeze-Drying Digital Twin",
    page_icon="❄️",
    layout="wide",
)

R_GAS = 8.314462618
TORR_TO_PA = 133.322368
CM2_TO_M2 = 1.0e-4
G_TO_KG = 1.0e-3
HR_TO_S = 3600.0


# ============================================================
# SHARED PHYSICAL FUNCTIONS
# ============================================================
def vapor_pressure_ice_torr(temperature_c):
    return np.exp(-6144.96 / (temperature_c + 273.15) + 24.01849)


def vapor_pressure_water_pa(temperature_k):
    temperature_c = temperature_k - 273.15
    return 611.21 * np.exp(
        (18.678 - temperature_c / 234.5)
        * temperature_c
        / (257.14 + temperature_c)
    )


def product_resistance(ldry_cm, parameters):
    return parameters["R0"] + (
        parameters["A1"] * ldry_cm
    ) / (1.0 + parameters["A2"] * ldry_cm)


def recipe_value(recipe, time_hr):
    matching = recipe[
        (recipe["Start (h)"] <= time_hr)
        & (time_hr < recipe["End (h)"])
    ]
    if matching.empty:
        matching = recipe.iloc[[-1]]
    row = matching.iloc[0]
    return float(row["Shelf T (°C)"]), float(row["Pressure (Torr)"])


def solve_tridiagonal(lower, diagonal, upper, rhs):
    n = len(rhs)
    corrected_upper = np.zeros(n - 1)
    corrected_rhs = np.zeros(n)

    corrected_upper[0] = upper[0] / diagonal[0]
    corrected_rhs[0] = rhs[0] / diagonal[0]

    for i in range(1, n - 1):
        denominator = diagonal[i] - lower[i - 1] * corrected_upper[i - 1]
        corrected_upper[i] = upper[i] / denominator
        corrected_rhs[i] = (
            rhs[i] - lower[i - 1] * corrected_rhs[i - 1]
        ) / denominator

    denominator = diagonal[-1] - lower[-1] * corrected_upper[-1]
    corrected_rhs[-1] = (
        rhs[-1] - lower[-1] * corrected_rhs[-2]
    ) / denominator

    solution = np.zeros(n)
    solution[-1] = corrected_rhs[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = corrected_rhs[i] - corrected_upper[i] * solution[i + 1]

    return solution


def equilibrium_moisture(activity, temperature_k, parameters):
    af = parameters["Af0"] * np.exp(parameters["Ef1"] / temperature_k)
    al = parameters["AL0"] * np.exp(parameters["Ef2"] / temperature_k)
    bl = parameters["BL0"] * np.exp(parameters["Ef3"] / temperature_k)
    value = (
        af * activity ** parameters["n_iso"]
        + al * activity / (1.0 + bl * activity)
    )
    return np.clip(value, 0.0, 1.0)


def desorption_rate_constant(temperature_k, parameters):
    return parameters["A_des"] * np.exp(
        -parameters["E_des"] / (R_GAS * temperature_k)
    )


def glass_transition_temperature(moisture, parameters):
    water_fraction = np.clip(moisture / (1.0 + moisture), 0.0, 0.5)
    dry_fraction = 1.0 - water_fraction
    base = (
        dry_fraction * parameters["Tg_dry"]
        + parameters["K_GT"] * water_fraction * parameters["Tg_water"]
    ) / (
        dry_fraction + parameters["K_GT"] * water_fraction
    )
    return (
        base
        + parameters["GT_a1"] * water_fraction * dry_fraction
        + parameters["GT_a2"] * water_fraction**2 * dry_fraction
    )


def default_recipe(phase):
    if phase == "Phase 3":
        return pd.DataFrame(
            {
                "Start (h)": [0.0, 10.0, 40.0, 55.0],
                "End (h)": [10.0, 40.0, 55.0, 120.0],
                "Shelf T (°C)": [-20.0, -8.0, 10.0, 30.0],
                "Pressure (Torr)": [0.10, 0.10, 0.08, 0.05],
            }
        )

    return pd.DataFrame(
        {
            "Start (h)": [0.0, 4.0, 5.0, 7.0, 12.0],
            "End (h)": [4.0, 5.0, 7.0, 12.0, 100.0],
            "Shelf T (°C)": [-8.0, -3.0, -3.0, -11.0, -8.0],
            "Pressure (Torr)": [0.10, 0.10, 0.15, 0.05, 0.10],
        }
    )


# ============================================================
# STREAMLIT INPUT BUILDERS
# ============================================================
def shared_inputs(prefix):
    st.sidebar.subheader("Geometry and formulation")

    fill_ml = st.sidebar.number_input(
        "Fill volume (mL)",
        min_value=0.01,
        max_value=500.0,
        value=48.0,
        step=1.0,
        key=prefix + "fill",
    )
    outer_diameter_cm = st.sidebar.number_input(
        "Outer diameter (cm)",
        min_value=0.01,
        max_value=20.0,
        value=4.7,
        step=0.1,
        key=prefix + "dout",
    )
    wall_thickness_cm = st.sidebar.number_input(
        "Wall thickness (cm)",
        min_value=0.0,
        max_value=5.0,
        value=0.17,
        step=0.01,
        key=prefix + "wall",
    )
    solid_fraction = st.sidebar.number_input(
        "Solid fraction",
        min_value=0.0,
        max_value=0.99,
        value=0.0625,
        step=0.0025,
        format="%.4f",
        key=prefix + "solid",
    )
    ice_density = st.sidebar.number_input(
        "Ice density (kg/m³)",
        min_value=1.0,
        max_value=2000.0,
        value=918.0,
        step=1.0,
        key=prefix + "rhoice",
    )
    sublimation_heat = st.sidebar.number_input(
        "Sublimation heat (kJ/kg)",
        min_value=1.0,
        max_value=10000.0,
        value=2834.6,
        step=1.0,
        key=prefix + "dH",
    ) * 1000.0
    critical_temperature = st.sidebar.number_input(
        "Critical product temperature (°C)",
        min_value=-100.0,
        max_value=100.0,
        value=-10.0,
        step=1.0,
        key=prefix + "Tcrit",
    )

    st.sidebar.subheader("Product resistance")
    r0 = st.sidebar.number_input(
        "R0 (cm²·h·Torr/g)",
        min_value=0.0001,
        max_value=100000.0,
        value=44.59,
        step=1.0,
        key=prefix + "R0",
    )
    a1 = st.sidebar.number_input(
        "A1",
        min_value=-100000.0,
        max_value=100000.0,
        value=1451.73,
        step=1.0,
        key=prefix + "A1",
    )
    a2 = st.sidebar.number_input(
        "A2",
        min_value=-10000.0,
        max_value=10000.0,
        value=12.93,
        step=0.1,
        key=prefix + "A2",
    )

    inner_diameter_cm = outer_diameter_cm - 2.0 * wall_thickness_cm
    if inner_diameter_cm <= 0:
        st.error("Outer diameter must exceed twice the wall thickness.")
        st.stop()

    area_cm2 = math.pi * inner_diameter_cm**2 / 4.0
    area_m2 = area_cm2 * CM2_TO_M2
    cake_height_m = (fill_ml / area_cm2) / 100.0

    return {
        "fill_ml": fill_ml,
        "outer_diameter_cm": outer_diameter_cm,
        "wall_thickness_cm": wall_thickness_cm,
        "inner_diameter_cm": inner_diameter_cm,
        "solid_fraction": solid_fraction,
        "rho_ice": ice_density,
        "delta_h_sub": sublimation_heat,
        "critical_temperature": critical_temperature,
        "R0": r0,
        "A1": a1,
        "A2": a2,
        "area_cm2": area_cm2,
        "area_m2": area_m2,
        "cake_height_m": cake_height_m,
    }


def material_inputs(prefix):
    with st.sidebar.expander("Paper material and boundary defaults"):
        rho_frozen = st.number_input(
            "Frozen density (kg/m³)",
            min_value=1.0,
            max_value=5000.0,
            value=927.1,
            step=1.0,
            key=prefix + "rhof",
        )
        cp_frozen = st.number_input(
            "Frozen Cp (J/kg/K)",
            min_value=1.0,
            max_value=10000.0,
            value=1940.0,
            step=10.0,
            key=prefix + "cpf",
        )
        k_frozen = st.number_input(
            "Frozen k (W/m/K)",
            min_value=0.000001,
            max_value=20.0,
            value=2.763,
            step=0.001,
            format="%.6f",
            key=prefix + "kf",
        )
        rho_dry = st.number_input(
            "Dry density (kg/m³)",
            min_value=0.01,
            max_value=5000.0,
            value=55.0,
            step=1.0,
            key=prefix + "rhod",
        )
        cp_dry = st.number_input(
            "Dry Cp (J/kg/K)",
            min_value=1.0,
            max_value=10000.0,
            value=259.5,
            step=1.0,
            key=prefix + "cpd",
        )
        k_dry_intercept = st.number_input(
            "Dry k intercept (W/m/K)",
            min_value=0.0000001,
            max_value=10.0,
            value=0.02706,
            step=0.00001,
            format="%.6f",
            key=prefix + "kd0",
        )
        k_dry_pressure = st.number_input(
            "Dry k pressure coefficient (W/m/K/Pa)",
            min_value=0.0,
            max_value=1.0,
            value=8.826e-5,
            step=1.0e-6,
            format="%.8f",
            key=prefix + "kdP",
        )
        h_top = st.number_input(
            "Top heat-transfer coefficient (W/m²/K)",
            min_value=0.0,
            max_value=1000.0,
            value=4.2,
            step=0.1,
            key=prefix + "htop",
        )
        h_gap = st.number_input(
            "Bottom gap coefficient (W/m²/K)",
            min_value=0.0,
            max_value=1000.0,
            value=8.73,
            step=0.1,
            key=prefix + "hgap",
        )
        h_contact = st.number_input(
            "Bottom contact coefficient (W/m²/K)",
            min_value=0.0,
            max_value=1000.0,
            value=16.22,
            step=0.1,
            key=prefix + "hcontact",
        )
        contact_fraction = st.number_input(
            "Bottom contact fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.278,
            step=0.01,
            key=prefix + "contact",
        )
        h_side = st.number_input(
            "Edge side coefficient (W/m²/K)",
            min_value=0.0,
            max_value=1000.0,
            value=3.514,
            step=0.1,
            key=prefix + "hside",
        )
        glass_conductivity = st.number_input(
            "Glass conductivity (W/m/K)",
            min_value=0.0001,
            max_value=100.0,
            value=1.1,
            step=0.1,
            key=prefix + "glassk",
        )
        condenser_capacity = st.number_input(
            "Condenser capacity per vial equivalent (kg/s)",
            min_value=0.0,
            max_value=1.0,
            value=0.0005,
            step=0.00001,
            format="%.7f",
            key=prefix + "capacity",
        )

    return {
        "rho_frozen": rho_frozen,
        "cp_frozen": cp_frozen,
        "k_frozen": k_frozen,
        "rho_dry": rho_dry,
        "cp_dry": cp_dry,
        "k_dry_intercept": k_dry_intercept,
        "k_dry_pressure": k_dry_pressure,
        "h_top": h_top,
        "h_gap": h_gap,
        "h_contact": h_contact,
        "contact_fraction": contact_fraction,
        "h_side": h_side,
        "glass_conductivity": glass_conductivity,
        "condenser_capacity": condenser_capacity,
    }


def moisture_inputs(prefix):
    with st.sidebar.expander("Moisture, sorption and Tg defaults"):
        initial_moisture = st.number_input(
            "Initial bound water (kg/kg)",
            min_value=0.0,
            max_value=10.0,
            value=0.2059,
            step=0.001,
            format="%.4f",
            key=prefix + "C0",
        )
        desorption_a = st.number_input(
            "Desorption A (1/s)",
            min_value=0.0,
            max_value=1.0e10,
            value=3340.0,
            step=10.0,
            key=prefix + "Ades",
        )
        desorption_e = st.number_input(
            "Desorption Ea (J/mol)",
            min_value=0.0,
            max_value=1.0e7,
            value=8136.0,
            step=10.0,
            key=prefix + "Edes",
        )
        af0 = st.number_input(
            "Af0",
            min_value=0.0,
            max_value=1.0,
            value=4.26e-5,
            step=1.0e-6,
            format="%.8f",
            key=prefix + "Af0",
        )
        ef1 = st.number_input(
            "Ef1",
            min_value=-1.0e6,
            max_value=1.0e6,
            value=1929.0,
            step=1.0,
            key=prefix + "Ef1",
        )
        n_iso = st.number_input(
            "Sorption exponent n",
            min_value=0.01,
            max_value=10.0,
            value=0.3,
            step=0.01,
            key=prefix + "n_iso",
        )
        al0 = st.number_input(
            "AL0",
            min_value=0.0,
            max_value=1.0,
            value=6.42e-5,
            step=1.0e-6,
            format="%.8f",
            key=prefix + "AL0",
        )
        bl0 = st.number_input(
            "BL0",
            min_value=0.0,
            max_value=1.0,
            value=7.88e-6,
            step=1.0e-6,
            format="%.8f",
            key=prefix + "BL0",
        )
        ef2 = st.number_input(
            "Ef2",
            min_value=-1.0e6,
            max_value=1.0e6,
            value=5028.0,
            step=1.0,
            key=prefix + "Ef2",
        )
        ef3 = st.number_input(
            "Ef3",
            min_value=-1.0e6,
            max_value=1.0e6,
            value=5028.0,
            step=1.0,
            key=prefix + "Ef3",
        )
        tg_dry = st.number_input(
            "Dry-solid Tg (K)",
            min_value=1.0,
            max_value=1000.0,
            value=348.2,
            step=1.0,
            key=prefix + "Tgd",
        )
        tg_water = st.number_input(
            "Water Tg (K)",
            min_value=1.0,
            max_value=1000.0,
            value=135.0,
            step=1.0,
            key=prefix + "Tgw",
        )
        k_gt = st.number_input(
            "Gordon-Taylor K",
            min_value=0.0001,
            max_value=100.0,
            value=0.092,
            step=0.001,
            key=prefix + "Kgt",
        )
        gt_a1 = st.number_input(
            "Modified GT a1",
            min_value=-10000.0,
            max_value=10000.0,
            value=480.8,
            step=1.0,
            key=prefix + "g1",
        )
        gt_a2 = st.number_input(
            "Modified GT a2",
            min_value=-10000.0,
            max_value=10000.0,
            value=1224.0,
            step=1.0,
            key=prefix + "g2",
        )

    return {
        "initial_moisture": initial_moisture,
        "A_des": desorption_a,
        "E_des": desorption_e,
        "Af0": af0,
        "Ef1": ef1,
        "n_iso": n_iso,
        "AL0": al0,
        "BL0": bl0,
        "Ef2": ef2,
        "Ef3": ef3,
        "Tg_dry": tg_dry,
        "Tg_water": tg_water,
        "K_GT": k_gt,
        "GT_a1": gt_a1,
        "GT_a2": gt_a2,
    }


# ============================================================
# PHASE 1 / PHASE 3 1D NONSTEADY SOLVER
# ============================================================
def solve_phase1(parameters, recipe, store_profiles=True):
    n = int(parameters["axial_nodes"])
    height = parameters["cake_height_m"]
    dz = height / (n - 1)
    z = np.linspace(0.0, height, n)
    temperature = np.full(n, parameters["initial_temperature"] + 273.15)
    moisture = np.full(n, parameters["initial_moisture"])
    dry_layer = 0.0
    ice_initial = (
        parameters["rho_ice"]
        * parameters["area_m2"]
        * height
        * (1.0 - parameters["solid_fraction"])
    )
    ice_mass = ice_initial
    elapsed = 0.0
    rows = []
    profiles = {}
    next_profile = 0.0
    endpoint = np.nan
    peak_temperature = -999.0
    capacity_exceeded = False

    max_steps = int(
        parameters["max_time_hr"] * 3600.0 / parameters["dt_s"]
    )

    for _ in range(max_steps):
        time_hr = elapsed / 3600.0
        shelf_c, pressure_torr = recipe_value(recipe, time_hr)
        dry_mask = z >= height - dry_layer

        conductivity = np.where(
            dry_mask,
            parameters["k_dry_intercept"]
            + parameters["k_dry_pressure"]
            * pressure_torr
            * TORR_TO_PA,
            parameters["k_frozen"],
        )
        density = np.where(
            dry_mask,
            parameters["rho_dry"],
            parameters["rho_frozen"],
        )
        heat_capacity = np.where(
            dry_mask,
            parameters["cp_dry"],
            parameters["cp_frozen"],
        )

        interface_index = int(
            np.argmin(np.abs(z - (height - dry_layer)))
        )
        interface_temperature = temperature[interface_index] - 273.15
        sublimation_g_hr = max(
            0.0,
            parameters["area_cm2"]
            * (vapor_pressure_ice_torr(interface_temperature) - pressure_torr)
            / max(product_resistance(dry_layer * 100.0, parameters), 1.0e-12),
        )
        sublimation_kg_s = min(
            sublimation_g_hr * G_TO_KG / HR_TO_S,
            ice_mass / parameters["dt_s"] if ice_mass > 0 else 0.0,
        )
        capacity_exceeded = (
            capacity_exceeded
            or sublimation_kg_s > parameters["condenser_capacity"]
        )
        latent_flux = (
            sublimation_kg_s
            / parameters["area_m2"]
            * parameters["delta_h_sub"]
        )

        lower = np.zeros(n - 1)
        diagonal = np.zeros(n)
        upper = np.zeros(n - 1)
        rhs = density * heat_capacity * temperature / parameters["dt_s"]

        h_bottom_raw = (
            parameters["contact_fraction"] * parameters["h_contact"]
            + (1.0 - parameters["contact_fraction"]) * parameters["h_gap"]
        )
        h_bottom = 1.0 / (
            1.0 / max(h_bottom_raw, 1.0e-12)
            + (parameters["wall_thickness_cm"] / 100.0)
            / max(parameters["glass_conductivity"], 1.0e-12)
        )

        for i in range(n):
            diagonal[i] = density[i] * heat_capacity[i] / parameters["dt_s"]

            if i > 0:
                k_w = (
                    2.0 * conductivity[i] * conductivity[i - 1]
                    / max(conductivity[i] + conductivity[i - 1], 1.0e-15)
                )
                coefficient = k_w / dz**2
                diagonal[i] += coefficient
                lower[i - 1] = -coefficient
            else:
                coefficient = h_bottom / dz
                diagonal[i] += coefficient
                rhs[i] += coefficient * (shelf_c + 273.15)

            if i < n - 1:
                k_e = (
                    2.0 * conductivity[i] * conductivity[i + 1]
                    / max(conductivity[i] + conductivity[i + 1], 1.0e-15)
                )
                coefficient = k_e / dz**2
                diagonal[i] += coefficient
                upper[i] = -coefficient
            else:
                coefficient = parameters["h_top"] / dz
                diagonal[i] += coefficient
                rhs[i] += coefficient * (
                    parameters["upper_surface_temperature"] + 273.15
                )

        rhs[interface_index] -= latent_flux / dz
        temperature = solve_tridiagonal(lower, diagonal, upper, rhs)

        water_activity = np.clip(
            pressure_torr
            * TORR_TO_PA
            / np.maximum(vapor_pressure_water_pa(temperature), 1.0e-12),
            0.0,
            0.999,
        )
        for i in np.where(dry_mask)[0]:
            equilibrium = equilibrium_moisture(
                water_activity[i], temperature[i], parameters
            )
            rate = desorption_rate_constant(temperature[i], parameters)
            moisture[i] = equilibrium + (
                moisture[i] - equilibrium
            ) * np.exp(-rate * parameters["dt_s"])

        removed = sublimation_kg_s * parameters["dt_s"]
        ice_mass = max(0.0, ice_mass - removed)
        dry_layer = min(
            height,
            dry_layer
            + removed
            / max(
                parameters["rho_ice"]
                * parameters["area_m2"]
                * (1.0 - parameters["solid_fraction"]),
                1.0e-15,
            ),
        )
        elapsed += parameters["dt_s"]

        temperature_c = temperature - 273.15
        tg = np.array(
            [glass_transition_temperature(value, parameters) for value in moisture]
        )
        margin = tg - temperature
        peak_temperature = max(peak_temperature, float(temperature_c.max()))
        drying_percent = 100.0 * (
            1.0 - ice_mass / max(ice_initial, 1.0e-15)
        )

        rows.append(
            {
                "Time (h)": elapsed / 3600.0,
                "Shelf T (°C)": shelf_c,
                "Pressure (Torr)": pressure_torr,
                "Bottom T (°C)": temperature_c[0],
                "Core T (°C)": temperature_c[n // 2],
                "Top T (°C)": temperature_c[-1],
                "Maximum product T (°C)": temperature_c.max(),
                "Interface T (°C)": temperature_c[interface_index],
                "Dry layer (cm)": dry_layer * 100.0,
                "Drying (%)": drying_percent,
                "Sublimation rate (g/h)": sublimation_g_hr,
                "Mean moisture": moisture.mean(),
                "Maximum moisture": moisture.max(),
                "Minimum Tg-T (°C)": margin.min(),
            }
        )

        if store_profiles and elapsed >= next_profile:
            profiles[round(elapsed / 3600.0, 3)] = pd.DataFrame(
                {
                    "Height (cm)": z * 100.0,
                    "Temperature (°C)": temperature_c,
                    "Moisture": moisture,
                    "Tg-T (°C)": margin,
                    "Region": np.where(dry_mask, "Dry", "Frozen"),
                }
            )
            next_profile += parameters["profile_interval_hr"] * 3600.0

        if ice_mass <= 1.0e-12 and np.isnan(endpoint):
            endpoint = elapsed / 3600.0

        if (
            ice_mass <= 1.0e-12
            and elapsed / 3600.0 >= recipe["End (h)"].max()
        ):
            break

    history = pd.DataFrame(rows)
    summary = {
        "Endpoint (h)": endpoint,
        "Peak T (°C)": peak_temperature,
        "Final drying (%)": history["Drying (%)"].iloc[-1],
        "Final moisture": history["Mean moisture"].iloc[-1],
        "Minimum Tg-T (°C)": history["Minimum Tg-T (°C)"].min(),
        "Capacity exceeded": capacity_exceeded,
    }
    return history, profiles, summary


# ============================================================
# PHASE 2 AXISYMMETRIC 2D SOLVER
# ============================================================
def flat_index(radial_index, axial_index, axial_nodes):
    return radial_index * axial_nodes + axial_index


def solve_phase2(parameters, recipe, store_maps=True):
    nr = int(parameters["radial_nodes"])
    nz = int(parameters["axial_nodes"])
    radius = parameters["radius_m"]
    height = parameters["cake_height_m"]
    dr = radius / (nr - 1)
    dz = height / (nz - 1)
    radial_positions = np.linspace(0.0, radius, nr)
    axial_positions = np.linspace(0.0, height, nz)
    temperature = np.full(
        (nr, nz), parameters["initial_temperature"] + 273.15
    )
    dry_layer = np.zeros(nr)
    initial_ice_mass = (
        parameters["rho_ice"]
        * math.pi
        * radius**2
        * height
        * (1.0 - parameters["solid_fraction"])
    )
    ice_mass = initial_ice_mass
    elapsed = 0.0
    rows = []
    maps = {}
    next_map = 0.0
    endpoint = np.nan
    peak_temperature = -999.0
    capacity_exceeded = False

    max_steps = int(
        parameters["max_time_hr"] * 3600.0 / parameters["dt_s"]
    )

    for _ in range(max_steps):
        time_hr = elapsed / 3600.0
        shelf_c, pressure_torr = recipe_value(recipe, time_hr)

        dry_mask = np.zeros((nr, nz), dtype=bool)
        for i in range(nr):
            dry_mask[i, :] = axial_positions >= height - dry_layer[i]

        conductivity = np.where(
            dry_mask,
            parameters["k_dry_intercept"]
            + parameters["k_dry_pressure"]
            * pressure_torr
            * TORR_TO_PA,
            parameters["k_frozen"],
        )
        density = np.where(
            dry_mask, parameters["rho_dry"], parameters["rho_frozen"]
        )
        heat_capacity = np.where(
            dry_mask, parameters["cp_dry"], parameters["cp_frozen"]
        )

        local_mass_flow = np.zeros(nr)
        latent_sink = np.zeros((nr, nz))

        for i in range(nr):
            interface_index = int(
                np.argmin(
                    np.abs(
                        axial_positions - (height - dry_layer[i])
                    )
                )
            )
            radius_in = max(0.0, radial_positions[i] - dr / 2.0)
            radius_out = min(radius, radial_positions[i] + dr / 2.0)
            annular_area = math.pi * (radius_out**2 - radius_in**2)
            sublimation_g_hr = max(
                0.0,
                annular_area
                / CM2_TO_M2
                * (
                    vapor_pressure_ice_torr(
                        temperature[i, interface_index] - 273.15
                    )
                    - pressure_torr
                )
                / max(
                    product_resistance(dry_layer[i] * 100.0, parameters),
                    1.0e-12,
                ),
            )
            local_mass_flow[i] = sublimation_g_hr * G_TO_KG / HR_TO_S
            latent_sink[i, interface_index] = (
                local_mass_flow[i]
                / max(annular_area, 1.0e-15)
                * parameters["delta_h_sub"]
                / dz
            )

        scale = (
            min(
                1.0,
                ice_mass
                / max(
                    local_mass_flow.sum() * parameters["dt_s"],
                    1.0e-30,
                ),
            )
            if ice_mass > 0
            else 0.0
        )
        local_mass_flow *= scale
        latent_sink *= scale
        capacity_exceeded = (
            capacity_exceeded
            or local_mass_flow.sum() > parameters["condenser_capacity"]
        )

        total_nodes = nr * nz
        matrix = lil_matrix((total_nodes, total_nodes))
        rhs = np.zeros(total_nodes)

        h_bottom_raw = (
            parameters["contact_fraction"] * parameters["h_contact"]
            + (1.0 - parameters["contact_fraction"]) * parameters["h_gap"]
        )
        h_bottom = 1.0 / (
            1.0 / max(h_bottom_raw, 1.0e-12)
            + (parameters["wall_thickness_cm"] / 100.0)
            / max(parameters["glass_conductivity"], 1.0e-12)
        )

        for i in range(nr):
            for j in range(nz):
                index = flat_index(i, j, nz)
                storage = (
                    density[i, j]
                    * heat_capacity[i, j]
                    / parameters["dt_s"]
                )
                matrix[index, index] = storage
                rhs[index] = storage * temperature[i, j] - latent_sink[i, j]

                if j > 0:
                    k_value = (
                        2.0
                        * conductivity[i, j]
                        * conductivity[i, j - 1]
                        / max(
                            conductivity[i, j] + conductivity[i, j - 1],
                            1.0e-15,
                        )
                    )
                    coefficient = k_value / dz**2
                    matrix[index, index] += coefficient
                    matrix[index, flat_index(i, j - 1, nz)] -= coefficient
                else:
                    coefficient = h_bottom / dz
                    matrix[index, index] += coefficient
                    rhs[index] += coefficient * (shelf_c + 273.15)

                if j < nz - 1:
                    k_value = (
                        2.0
                        * conductivity[i, j]
                        * conductivity[i, j + 1]
                        / max(
                            conductivity[i, j] + conductivity[i, j + 1],
                            1.0e-15,
                        )
                    )
                    coefficient = k_value / dz**2
                    matrix[index, index] += coefficient
                    matrix[index, flat_index(i, j + 1, nz)] -= coefficient
                else:
                    coefficient = parameters["h_top"] / dz
                    matrix[index, index] += coefficient
                    rhs[index] += coefficient * (
                        parameters["upper_surface_temperature"] + 273.15
                    )

                if i > 0:
                    k_value = (
                        2.0
                        * conductivity[i, j]
                        * conductivity[i - 1, j]
                        / max(
                            conductivity[i, j] + conductivity[i - 1, j],
                            1.0e-15,
                        )
                    )
                    coefficient = k_value / dr**2
                    matrix[index, index] += coefficient
                    matrix[index, flat_index(i - 1, j, nz)] -= coefficient

                if i < nr - 1:
                    k_value = (
                        2.0
                        * conductivity[i, j]
                        * conductivity[i + 1, j]
                        / max(
                            conductivity[i, j] + conductivity[i + 1, j],
                            1.0e-15,
                        )
                    )
                    coefficient = k_value / dr**2
                    matrix[index, index] += coefficient
                    matrix[index, flat_index(i + 1, j, nz)] -= coefficient
                else:
                    coefficient = parameters["active_side_h"] / dr
                    matrix[index, index] += coefficient
                    rhs[index] += coefficient * (
                        parameters["wall_temperature"] + 273.15
                    )

        temperature = spsolve(matrix.tocsr(), rhs).reshape(nr, nz)

        removed_by_radius = local_mass_flow * parameters["dt_s"]
        ice_mass = max(0.0, ice_mass - removed_by_radius.sum())

        for i in range(nr):
            radius_in = max(0.0, radial_positions[i] - dr / 2.0)
            radius_out = min(radius, radial_positions[i] + dr / 2.0)
            annular_area = math.pi * (radius_out**2 - radius_in**2)
            dry_layer[i] = min(
                height,
                dry_layer[i]
                + removed_by_radius[i]
                / max(
                    parameters["rho_ice"]
                    * annular_area
                    * (1.0 - parameters["solid_fraction"]),
                    1.0e-15,
                ),
            )

        elapsed += parameters["dt_s"]
        temperature_c = temperature - 273.15
        peak_temperature = max(peak_temperature, float(temperature_c.max()))
        drying_percent = 100.0 * (
            1.0 - ice_mass / max(initial_ice_mass, 1.0e-15)
        )

        rows.append(
            {
                "Time (h)": elapsed / 3600.0,
                "Shelf T (°C)": shelf_c,
                "Pressure (Torr)": pressure_torr,
                "Center core T (°C)": temperature_c[0, nz // 2],
                "Wall core T (°C)": temperature_c[-1, nz // 2],
                "Maximum T (°C)": temperature_c.max(),
                "Center dry layer (cm)": dry_layer[0] * 100.0,
                "Wall dry layer (cm)": dry_layer[-1] * 100.0,
                "Mean dry layer (cm)": dry_layer.mean() * 100.0,
                "Drying (%)": drying_percent,
                "Vial load (kg/s)": local_mass_flow.sum(),
            }
        )

        if store_maps and elapsed >= next_map:
            maps[round(elapsed / 3600.0, 3)] = {
                "temperature": temperature_c.copy(),
                "dry_layer": dry_layer.copy(),
                "radius": radial_positions.copy(),
                "height": axial_positions.copy(),
            }
            next_map += parameters["profile_interval_hr"] * 3600.0

        if ice_mass <= 1.0e-12 and np.isnan(endpoint):
            endpoint = elapsed / 3600.0

        if (
            ice_mass <= 1.0e-12
            and elapsed / 3600.0 >= recipe["End (h)"].max()
        ):
            break

    history = pd.DataFrame(rows)
    summary = {
        "Endpoint (h)": endpoint,
        "Peak T (°C)": peak_temperature,
        "Final drying (%)": history["Drying (%)"].iloc[-1],
        "Capacity exceeded": capacity_exceeded,
    }
    return history, maps, summary


# ============================================================
# APP SHELL
# ============================================================
st.title("Integrated Freeze-Drying Simulation Suite")
st.caption(
    "One application containing Phase 1 nonsteady 1D, Phase 2 axisymmetric "
    "2D, and Phase 3 primary-secondary quality simulations."
)

phase = st.sidebar.radio(
    "Select simulation module",
    ["Overview", "Phase 1", "Phase 2", "Phase 3"],
)

if phase == "Overview":
    st.markdown(
        """
## Modules

- **Phase 1:** axial nonsteady heat transfer, planar moving interface, moisture and Tg tracking.
- **Phase 2:** radial-axial temperature field, radius-dependent curved interface, robustness assessment.
- **Phase 3:** integrated primary and secondary drying, residual-moisture uniformity, Tg safety, and quality endpoint assessment.

All reference-paper defaults remain editable. Validate formulation-specific resistance, heat-transfer, sorption, desorption, and glass-transition parameters before process decisions.
"""
    )
    st.stop()

prefix = phase.replace(" ", "_") + "_"
parameters = shared_inputs(prefix)
parameters.update(material_inputs(prefix))

if phase in ("Phase 1", "Phase 3"):
    parameters.update(moisture_inputs(prefix))

with st.sidebar.expander("Numerical settings"):
    if phase == "Phase 2":
        parameters["radial_nodes"] = st.number_input(
            "Radial nodes",
            min_value=5,
            max_value=31,
            value=11,
            step=2,
            key="phase2_radial_nodes",
        )
        parameters["axial_nodes"] = st.number_input(
            "Axial nodes",
            min_value=9,
            max_value=51,
            value=21,
            step=2,
            key="phase2_axial_nodes",
        )
    else:
        parameters["axial_nodes"] = st.number_input(
            "Axial nodes",
            min_value=11,
            max_value=101,
            value=31,
            step=2,
            key=prefix + "axial_nodes",
        )

    parameters["dt_s"] = st.number_input(
        "Time step (s)",
        min_value=1.0,
        max_value=3600.0,
        value=60.0 if phase == "Phase 2" else 30.0,
        step=1.0,
        key=prefix + "dt",
    )
    parameters["initial_temperature"] = st.number_input(
        "Initial product temperature (°C)",
        min_value=-100.0,
        max_value=30.0,
        value=-40.0,
        step=1.0,
        key=prefix + "initial_temperature",
    )
    parameters["upper_surface_temperature"] = st.number_input(
        "Upper surface temperature (°C)",
        min_value=-100.0,
        max_value=100.0,
        value=-20.0,
        step=1.0,
        key=prefix + "upper_temperature",
    )
    parameters["wall_temperature"] = st.number_input(
        "Chamber wall temperature (°C)",
        min_value=-100.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key=prefix + "wall_temperature",
    )
    parameters["max_time_hr"] = st.number_input(
        "Maximum simulation time (h)",
        min_value=1.0,
        max_value=500.0,
        value=120.0 if phase == "Phase 3" else 100.0,
        step=1.0,
        key=prefix + "max_time",
    )
    parameters["profile_interval_hr"] = st.number_input(
        "Profile/map interval (h)",
        min_value=0.1,
        max_value=100.0,
        value=5.0 if phase == "Phase 2" else 4.0,
        step=0.5,
        key=prefix + "profile_interval",
    )

if phase == "Phase 2":
    parameters["radius_m"] = parameters["inner_diameter_cm"] / 200.0
    parameters["active_side_h"] = st.sidebar.number_input(
        "Active side heat-transfer coefficient (W/m²/K)",
        min_value=0.0,
        max_value=1000.0,
        value=parameters["h_side"],
        step=0.1,
        key="phase2_active_side_h",
    )

if phase == "Phase 3":
    st.sidebar.subheader("Quality criteria")
    parameters["target_moisture"] = st.sidebar.number_input(
        "Target mean moisture (kg/kg)",
        min_value=0.0,
        max_value=10.0,
        value=0.01,
        step=0.001,
        format="%.4f",
        key="phase3_target_moisture",
    )
    parameters["maximum_moisture_rsd"] = st.sidebar.number_input(
        "Maximum moisture RSD (%)",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        key="phase3_maximum_rsd",
    )
    parameters["minimum_tg_margin"] = st.sidebar.number_input(
        "Minimum Tg-T margin (°C)",
        min_value=-200.0,
        max_value=500.0,
        value=5.0,
        step=1.0,
        key="phase3_minimum_tg_margin",
    )

st.subheader(f"{phase} process recipe")
recipe = st.data_editor(
    default_recipe(phase),
    num_rows="dynamic",
    use_container_width=True,
    key="recipe_" + phase,
)
recipe = recipe.sort_values("Start (h)").reset_index(drop=True)

geometry_columns = st.columns(4)
geometry_columns[0].metric(
    "Inner diameter", f"{parameters['inner_diameter_cm']:.3f} cm"
)
geometry_columns[1].metric("Product area", f"{parameters['area_cm2']:.3f} cm²")
geometry_columns[2].metric(
    "Cake height", f"{parameters['cake_height_m'] * 100.0:.3f} cm"
)
geometry_columns[3].metric("Selected module", phase)

run_simulation = st.button(f"Run {phase}", type="primary")
result_key = "result_" + phase

if run_simulation:
    with st.spinner(f"Running {phase} simulation..."):
        try:
            if phase == "Phase 2":
                result = solve_phase2(parameters, recipe)
            else:
                result = solve_phase1(parameters, recipe)
            st.session_state[result_key] = (
                result,
                parameters.copy(),
                recipe.copy(),
            )
        except Exception as exception:
            st.exception(exception)

if result_key not in st.session_state:
    st.info(f"Configure inputs and select **Run {phase}**.")
    st.stop()

(history, spatial_results, summary), used_parameters, used_recipe = (
    st.session_state[result_key]
)


# ============================================================
# PHASE-SPECIFIC RESULTS
# ============================================================
if phase == "Phase 1":
    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Endpoint",
        "Not reached"
        if np.isnan(summary["Endpoint (h)"])
        else f"{summary['Endpoint (h)']:.2f} h",
    )
    metric_columns[1].metric("Peak T", f"{summary['Peak T (°C)']:.2f} °C")
    metric_columns[2].metric(
        "Final drying", f"{summary['Final drying (%)']:.2f}%"
    )
    metric_columns[3].metric(
        "Final moisture", f"{summary['Final moisture']:.4f}"
    )
    metric_columns[4].metric(
        "Minimum Tg-T", f"{summary['Minimum Tg-T (°C)']:.2f} °C"
    )
    metric_columns[5].metric(
        "Capacity",
        "Exceeded" if summary["Capacity exceeded"] else "Within limit",
    )

    dashboard_tab, profile_tab, assumption_tab, data_tab = st.tabs(
        ["Dashboard", "Depth profiles", "Assumptions", "Data"]
    )

    with dashboard_tab:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        time = history["Time (h)"]
        axes[0, 0].plot(time, history["Shelf T (°C)"], "--", label="Shelf")
        axes[0, 0].plot(time, history["Bottom T (°C)"], label="Bottom")
        axes[0, 0].plot(time, history["Core T (°C)"], label="Core")
        axes[0, 0].plot(time, history["Top T (°C)"], label="Top")
        axes[0, 0].axhline(
            used_parameters["critical_temperature"], color="red", linestyle="--"
        )
        axes[0, 0].set_title("Temperature")
        axes[0, 0].legend()
        axes[0, 1].plot(time, history["Dry layer (cm)"])
        axes[0, 1].set_title("Moving interface")
        axes[1, 0].plot(time, history["Drying (%)"])
        axes[1, 0].set_title("Drying progression")
        axes[1, 1].plot(time, history["Sublimation rate (g/h)"])
        axes[1, 1].set_title("Sublimation rate")
        for axis in axes.flat:
            axis.set_xlabel("Time (h)")
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with profile_tab:
        profile_times = list(spatial_results.keys())
        selected_times = st.multiselect(
            "Profile times (h)",
            profile_times,
            default=profile_times[-min(4, len(profile_times)):],
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for time_key in selected_times:
            profile = spatial_results[time_key]
            axes[0].plot(
                profile["Temperature (°C)"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
            axes[1].plot(
                profile["Moisture"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
            axes[2].plot(
                profile["Tg-T (°C)"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
        for axis, title in zip(
            axes, ["Temperature", "Moisture", "Tg-T margin"]
        ):
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with assumption_tab:
        st.warning(
            "Phase 1 is a one-dimensional finite-volume moving-interface "
            "implementation using editable reference-paper defaults."
        )

    with data_tab:
        st.dataframe(history, use_container_width=True, height=500)

elif phase == "Phase 2":
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Endpoint",
        "Not reached"
        if np.isnan(summary["Endpoint (h)"])
        else f"{summary['Endpoint (h)']:.2f} h",
    )
    metric_columns[1].metric("Peak T", f"{summary['Peak T (°C)']:.2f} °C")
    metric_columns[2].metric(
        "Final drying", f"{summary['Final drying (%)']:.2f}%"
    )
    metric_columns[3].metric(
        "Capacity",
        "Exceeded" if summary["Capacity exceeded"] else "Within limit",
    )

    dashboard_tab, map_tab, interface_tab, assumption_tab, data_tab = st.tabs(
        ["Dashboard", "2D map", "Interface shapes", "Assumptions", "Data"]
    )

    with dashboard_tab:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        time = history["Time (h)"]
        axes[0, 0].plot(time, history["Shelf T (°C)"], "--", label="Shelf")
        axes[0, 0].plot(
            time, history["Center core T (°C)"], label="Center core"
        )
        axes[0, 0].plot(
            time, history["Wall core T (°C)"], label="Wall core"
        )
        axes[0, 0].axhline(
            used_parameters["critical_temperature"], color="red", linestyle="--"
        )
        axes[0, 0].set_title("Temperature")
        axes[0, 0].legend()
        axes[0, 1].plot(
            time, history["Center dry layer (cm)"], label="Center"
        )
        axes[0, 1].plot(
            time, history["Wall dry layer (cm)"], label="Wall"
        )
        axes[0, 1].set_title("Curved interface progression")
        axes[0, 1].legend()
        axes[1, 0].plot(time, history["Drying (%)"])
        axes[1, 0].set_title("Drying progression")
        axes[1, 1].plot(time, history["Vial load (kg/s)"])
        axes[1, 1].set_title("Vial vapor load")
        for axis in axes.flat:
            axis.set_xlabel("Time (h)")
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with map_tab:
        map_times = list(spatial_results.keys())
        selected_time = st.select_slider(
            "Map time (h)",
            options=map_times,
            value=map_times[-1],
        )
        map_data = spatial_results[selected_time]
        fig, axis = plt.subplots(figsize=(10, 6))
        contour = axis.contourf(
            map_data["radius"] * 100.0,
            map_data["height"] * 100.0,
            map_data["temperature"].T,
            25,
            cmap="coolwarm",
        )
        fig.colorbar(contour, ax=axis, label="Temperature (°C)")
        axis.plot(
            map_data["radius"] * 100.0,
            (used_parameters["cake_height_m"] - map_data["dry_layer"])
            * 100.0,
            color="black",
            linewidth=2,
            label="Sublimation front",
        )
        axis.set_xlabel("Radius (cm)")
        axis.set_ylabel("Height (cm)")
        axis.set_title(f"Temperature field at {selected_time:g} h")
        axis.legend()
        st.pyplot(fig)
        plt.close(fig)

    with interface_tab:
        fig, axis = plt.subplots(figsize=(10, 6))
        for time_key in list(spatial_results.keys())[-min(6, len(spatial_results)):]:
            map_data = spatial_results[time_key]
            axis.plot(
                map_data["radius"] * 100.0,
                (used_parameters["cake_height_m"] - map_data["dry_layer"])
                * 100.0,
                label=f"{time_key:g} h",
            )
        axis.set_xlabel("Radius (cm)")
        axis.set_ylabel("Interface height from vial bottom (cm)")
        axis.set_title("Radius-dependent sublimation interface")
        axis.grid(True, alpha=0.3)
        axis.legend()
        st.pyplot(fig)
        plt.close(fig)

    with assumption_tab:
        st.warning(
            "Phase 2 is a structured axisymmetric finite-volume/FEM-style "
            "solver. It resolves radial and axial fields but is not a fully "
            "remeshed ALE finite-element implementation."
        )

    with data_tab:
        st.dataframe(history, use_container_width=True, height=500)

else:
    final_profile = spatial_results[list(spatial_results.keys())[-1]]
    moisture_rsd = (
        100.0
        * final_profile["Moisture"].std()
        / max(final_profile["Moisture"].mean(), 1.0e-12)
    )
    endpoint_rows = history[
        (history["Drying (%)"] >= 99.999)
        & (history["Mean moisture"] <= used_parameters["target_moisture"])
    ]
    moisture_endpoint = (
        np.nan if endpoint_rows.empty else endpoint_rows["Time (h)"].iloc[0]
    )

    metric_columns = st.columns(6)
    metric_columns[0].metric(
        "Primary endpoint",
        "Not reached"
        if np.isnan(summary["Endpoint (h)"])
        else f"{summary['Endpoint (h)']:.2f} h",
    )
    metric_columns[1].metric(
        "Moisture endpoint",
        "Not reached"
        if np.isnan(moisture_endpoint)
        else f"{moisture_endpoint:.2f} h",
    )
    metric_columns[2].metric(
        "Final moisture", f"{summary['Final moisture']:.4f}"
    )
    metric_columns[3].metric("Moisture RSD", f"{moisture_rsd:.2f}%")
    metric_columns[4].metric(
        "Minimum Tg-T", f"{summary['Minimum Tg-T (°C)']:.2f} °C"
    )
    metric_columns[5].metric("Peak T", f"{summary['Peak T (°C)']:.2f} °C")

    dashboard_tab, quality_tab, profile_tab, assumption_tab, data_tab = st.tabs(
        ["Dashboard", "Quality endpoint", "Spatial quality", "Assumptions", "Data"]
    )

    with dashboard_tab:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        time = history["Time (h)"]
        axes[0, 0].plot(time, history["Shelf T (°C)"], "--", label="Shelf")
        axes[0, 0].plot(time, history["Core T (°C)"], label="Core")
        axes[0, 0].plot(time, history["Top T (°C)"], label="Top")
        axes[0, 0].set_title("Temperature")
        axes[0, 0].legend()
        axes[0, 1].plot(time, history["Drying (%)"])
        axes[0, 1].set_title("Primary drying")
        axes[1, 0].plot(time, history["Mean moisture"], label="Mean")
        axes[1, 0].plot(
            time, history["Maximum moisture"], "--", label="Maximum"
        )
        axes[1, 0].axhline(
            used_parameters["target_moisture"], color="green", linestyle="--"
        )
        axes[1, 0].set_title("Residual moisture")
        axes[1, 0].legend()
        axes[1, 1].plot(time, history["Minimum Tg-T (°C)"])
        axes[1, 1].axhline(
            used_parameters["minimum_tg_margin"],
            color="red",
            linestyle="--",
        )
        axes[1, 1].set_title("Tg-T safety margin")
        for axis in axes.flat:
            axis.set_xlabel("Time (h)")
            axis.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with quality_tab:
        quality_passed = (
            summary["Final moisture"] <= used_parameters["target_moisture"]
            and moisture_rsd <= used_parameters["maximum_moisture_rsd"]
            and summary["Minimum Tg-T (°C)"]
            >= used_parameters["minimum_tg_margin"]
        )
        if quality_passed:
            st.success("Quality endpoint satisfied.")
        else:
            st.warning("Quality endpoint not satisfied.")

        st.dataframe(
            pd.DataFrame(
                [
                    [
                        "Mean moisture",
                        summary["Final moisture"],
                        used_parameters["target_moisture"],
                        "≤",
                    ],
                    [
                        "Moisture RSD (%)",
                        moisture_rsd,
                        used_parameters["maximum_moisture_rsd"],
                        "≤",
                    ],
                    [
                        "Minimum Tg-T (°C)",
                        summary["Minimum Tg-T (°C)"],
                        used_parameters["minimum_tg_margin"],
                        "≥",
                    ],
                ],
                columns=["Attribute", "Result", "Criterion", "Operator"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    with profile_tab:
        profile_times = list(spatial_results.keys())
        selected_times = st.multiselect(
            "Profile times (h)",
            profile_times,
            default=profile_times[-min(4, len(profile_times)):],
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for time_key in selected_times:
            profile = spatial_results[time_key]
            axes[0].plot(
                profile["Temperature (°C)"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
            axes[1].plot(
                profile["Moisture"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
            axes[2].plot(
                profile["Tg-T (°C)"],
                profile["Height (cm)"],
                label=f"{time_key:g} h",
            )
        for axis, title in zip(
            axes, ["Temperature", "Moisture", "Tg-T margin"]
        ):
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with assumption_tab:
        st.warning(
            "Sorption, desorption, and modified Gordon-Taylor parameters are "
            "formulation-specific and require calibration against moisture and "
            "cake-quality data."
        )

    with data_tab:
        st.dataframe(history, use_container_width=True, height=500)


# ============================================================
# SHARED EXPORT
# ============================================================
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    history.to_excel(writer, sheet_name="History", index=False)
    used_recipe.to_excel(writer, sheet_name="Recipe", index=False)
    pd.DataFrame(
        summary.items(), columns=["Metric", "Value"]
    ).to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(
        [(key, str(value)) for key, value in used_parameters.items()],
        columns=["Parameter", "Value"],
    ).to_excel(writer, sheet_name="Inputs", index=False)

st.download_button(
    f"Download {phase} complete Excel",
    data=excel_buffer.getvalue(),
    file_name=f"{phase.lower().replace(' ', '_')}_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

st.divider()
st.caption(
    "Research model. Validate formulation-specific Rp, heat-transfer, sorption, "
    "desorption, moisture, Tg, endpoint, and equipment-capacity parameters before "
    "process-development decisions."
)
