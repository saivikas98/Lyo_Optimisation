import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import fsolve

st.set_page_config(
    page_title="Primary Drying Simulator",
    page_icon="❄️",
    layout="wide",
)

st.title("Primary Drying Center vs Edge Vial Simulator")
st.caption(
    "Transient primary-drying model with critical-product-temperature control, "
    "center and edge vial heat-transfer correlations, plots, and downloadable results."
)


# ============================================================
# MODEL FUNCTIONS
# ============================================================
def product_resistance(ldry, a5, a6, a7):
    return a5 + (a6 * ldry) / (1.0 + a7 * ldry)


def kv_center(pc):
    return (
        9.211 + (0.066 * pc * 1000.0) / (1.0 + 0.002 * pc * 1000.0)
    ) * 0.001434


def kv_edge(pc):
    return (
        (0.981 * pc * 1000.0) / (1.0 + 0.02 * pc * 1000.0)
    ) * 0.001434


def vapor_pressure_ice(temperature_c):
    temperature_k = temperature_c + 273.15
    return np.exp(-6144.96 / temperature_k + 24.01849)


def residual_transient(
    x,
    ldry,
    shelf_temperature,
    chamber_pressure,
    previous_product_temperature,
    dt,
    kv_function,
    product_area,
    vial_area,
    initial_cake_height,
    ice_density,
    solid_fraction,
    heat_capacity,
    ice_conductivity,
    sublimation_enthalpy,
    a5,
    a6,
    a7,
):
    interface_temperature, product_temperature = x
    ice_pressure = vapor_pressure_ice(interface_temperature)

    if ice_pressure <= chamber_pressure:
        return [1.0e6, 1.0e6]

    resistance = product_resistance(ldry, a5, a6, a7)
    sublimation_rate = max(
        0.0,
        (product_area / resistance) * (ice_pressure - chamber_pressure),
    )

    remaining_ice_height = max(initial_cake_height - ldry, 1.0e-6)
    thermal_mass = max(
        ice_density * product_area * ldry * solid_fraction,
        1.0e-8,
    )
    kv_value = kv_function(chamber_pressure)

    heat_input = kv_value * vial_area * (
        shelf_temperature - product_temperature
    )
    heat_sublimation = sublimation_rate * sublimation_enthalpy
    heat_accumulation = (
        thermal_mass
        * heat_capacity
        * (product_temperature - previous_product_temperature)
        / dt
    )

    energy_residual = heat_input - (
        heat_sublimation + heat_accumulation
    )
    conduction = (
        ice_conductivity
        * vial_area
        * (product_temperature - interface_temperature)
        / remaining_ice_height
    )
    conduction_residual = heat_input - conduction

    return [energy_residual, conduction_residual]


def solve_position(
    ldry,
    shelf_temperature,
    pressure,
    previous_product_temperature,
    interface_guess,
    dt,
    kv_function,
    parameters,
):
    solution, info, ier, message = fsolve(
        residual_transient,
        [interface_guess, previous_product_temperature],
        args=(
            ldry,
            shelf_temperature,
            pressure,
            previous_product_temperature,
            dt,
            kv_function,
            parameters["product_area"],
            parameters["vial_area"],
            parameters["initial_cake_height"],
            parameters["ice_density"],
            parameters["solid_fraction"],
            parameters["heat_capacity"],
            parameters["ice_conductivity"],
            parameters["sublimation_enthalpy"],
            parameters["a5"],
            parameters["a6"],
            parameters["a7"],
        ),
        full_output=True,
        xtol=1.0e-8,
        maxfev=300,
    )

    if ier != 1 or not np.all(np.isfinite(solution)):
        raise RuntimeError(f"Nonlinear solver did not converge: {message}")

    interface_temperature, product_temperature = solution
    resistance = product_resistance(
        ldry, parameters["a5"], parameters["a6"], parameters["a7"]
    )
    sublimation_rate = max(
        0.0,
        (parameters["product_area"] / resistance)
        * (vapor_pressure_ice(interface_temperature) - pressure),
    )

    return interface_temperature, product_temperature, sublimation_rate


def simulate(parameters):
    product_area = parameters["product_area"]
    vial_area = parameters["vial_area"]
    initial_cake_height = parameters["initial_cake_height"]
    dt = parameters["dt"]
    pressure = parameters["pressure"]

    shelf_temperature = (
        6144.96 / (24.01849 - math.log(pressure)) - 273.15
    )

    time_seconds = 0.0
    center_dry_layer = 0.0
    edge_dry_layer = 0.0
    center_interface_guess = parameters["initial_interface_temperature"]
    edge_interface_guess = parameters["initial_interface_temperature"]
    center_product_history = []
    edge_product_history = []
    center_interface_history = []
    edge_interface_history = []
    records = []
    rejected_steps = 0
    solver_failures = 0
    maximum_steps = int(parameters["max_duration_min"] * 60.0 / dt)

    for step in range(maximum_steps):
        center_previous_temperature = (
            center_product_history[-1]
            if center_product_history
            else center_interface_guess
        )
        edge_previous_temperature = (
            edge_product_history[-1]
            if edge_product_history
            else edge_interface_guess
        )

        try:
            if center_dry_layer < initial_cake_height:
                center_interface, center_product, center_rate = solve_position(
                    center_dry_layer,
                    shelf_temperature,
                    pressure,
                    center_previous_temperature,
                    center_interface_guess,
                    dt,
                    kv_center,
                    parameters,
                )
                center_increment = (
                    center_rate
                    * dt
                    / (
                        parameters["ice_density"]
                        * product_area
                        * (1.0 - parameters["solid_fraction"])
                    )
                )
                proposed_center_dry_layer = min(
                    initial_cake_height,
                    center_dry_layer + center_increment,
                )
            else:
                center_thermal_mass = (
                    parameters["ice_density"]
                    * product_area
                    * initial_cake_height
                    * parameters["solid_fraction"]
                )
                center_product = center_previous_temperature
                center_product += (
                    kv_center(pressure)
                    * vial_area
                    * (shelf_temperature - center_product)
                    * dt
                    / (center_thermal_mass * parameters["heat_capacity"])
                )
                center_interface = (
                    center_interface_history[-1]
                    if center_interface_history
                    else center_interface_guess
                )
                center_rate = 0.0
                proposed_center_dry_layer = center_dry_layer

            if edge_dry_layer < initial_cake_height:
                edge_interface, edge_product, edge_rate = solve_position(
                    edge_dry_layer,
                    shelf_temperature,
                    pressure,
                    edge_previous_temperature,
                    edge_interface_guess,
                    dt,
                    kv_edge,
                    parameters,
                )
                edge_increment = (
                    edge_rate
                    * dt
                    / (
                        parameters["ice_density"]
                        * product_area
                        * (1.0 - parameters["solid_fraction"])
                    )
                )
                proposed_edge_dry_layer = min(
                    initial_cake_height,
                    edge_dry_layer + edge_increment,
                )
            else:
                edge_thermal_mass = (
                    parameters["ice_density"]
                    * product_area
                    * initial_cake_height
                    * parameters["solid_fraction"]
                )
                edge_product = edge_previous_temperature
                edge_product += (
                    kv_edge(pressure)
                    * vial_area
                    * (shelf_temperature - edge_product)
                    * dt
                    / (edge_thermal_mass * parameters["heat_capacity"])
                )
                edge_interface = (
                    edge_interface_history[-1]
                    if edge_interface_history
                    else edge_interface_guess
                )
                edge_rate = 0.0
                proposed_edge_dry_layer = edge_dry_layer
        except RuntimeError:
            solver_failures += 1
            shelf_temperature -= parameters["constraint_step"]
            if solver_failures > parameters["max_solver_failures"]:
                raise RuntimeError(
                    "The nonlinear solver repeatedly failed. Review units, "
                    "initial guesses, pressure, and heat-transfer parameters."
                )
            continue

        center_interface = min(center_interface, shelf_temperature - 1.0)
        center_product = min(center_product, shelf_temperature - 1.0)
        edge_interface = min(edge_interface, shelf_temperature - 1.0)
        edge_product = min(edge_product, shelf_temperature - 1.0)

        active_product_temperatures = []
        if proposed_center_dry_layer < initial_cake_height:
            active_product_temperatures.append(center_product)
        if proposed_edge_dry_layer < initial_cake_height:
            active_product_temperatures.append(edge_product)

        if (
            active_product_temperatures
            and max(active_product_temperatures) > parameters["critical_temperature"]
        ):
            shelf_temperature -= parameters["constraint_step"]
            rejected_steps += 1
            if rejected_steps > parameters["max_rejected_steps"]:
                raise RuntimeError(
                    "Too many temperature-constraint rejections. "
                    "Review Tcrit, shelf ramp, or model units."
                )
            continue

        center_dry_layer = proposed_center_dry_layer
        edge_dry_layer = proposed_edge_dry_layer
        shelf_temperature = min(
            shelf_temperature, parameters["maximum_shelf_temperature"]
        )
        time_seconds += dt

        center_interface_history.append(center_interface)
        center_product_history.append(center_product)
        edge_interface_history.append(edge_interface)
        edge_product_history.append(edge_product)

        center_remaining_height = max(
            initial_cake_height - center_dry_layer, 0.0
        )
        edge_remaining_height = max(
            initial_cake_height - edge_dry_layer, 0.0
        )
        center_drying_percent = min(
            100.0, 100.0 * center_dry_layer / initial_cake_height
        )
        edge_drying_percent = min(
            100.0, 100.0 * edge_dry_layer / initial_cake_height
        )

        center_heat_input = (
            kv_center(pressure)
            * vial_area
            * (shelf_temperature - center_product)
        )
        edge_heat_input = (
            kv_edge(pressure)
            * vial_area
            * (shelf_temperature - edge_product)
        )
        center_conduction = (
            parameters["ice_conductivity"]
            * vial_area
            * (center_product - center_interface)
            / max(center_remaining_height, 1.0e-6)
        )
        edge_conduction = (
            parameters["ice_conductivity"]
            * vial_area
            * (edge_product - edge_interface)
            / max(edge_remaining_height, 1.0e-6)
        )

        records.append(
            {
                "Time (s)": time_seconds,
                "Time (min)": time_seconds / 60.0,
                "Shelf temperature (°C)": shelf_temperature,
                "Center interface temperature (°C)": center_interface,
                "Center product temperature (°C)": center_product,
                "Center dry layer (cm)": center_dry_layer,
                "Center remaining cake (cm)": center_remaining_height,
                "Center drying (%)": center_drying_percent,
                "Center sublimation rate": center_rate,
                "Center heat input": center_heat_input,
                "Center conduction": center_conduction,
                "Edge interface temperature (°C)": edge_interface,
                "Edge product temperature (°C)": edge_product,
                "Edge dry layer (cm)": edge_dry_layer,
                "Edge remaining cake (cm)": edge_remaining_height,
                "Edge drying (%)": edge_drying_percent,
                "Edge sublimation rate": edge_rate,
                "Edge heat input": edge_heat_input,
                "Edge conduction": edge_conduction,
            }
        )

        if shelf_temperature < -10.0:
            shelf_temperature += parameters["low_temperature_ramp"] * dt / 60.0
        else:
            shelf_temperature += parameters["high_temperature_ramp"] * dt / 60.0

        center_interface_guess = center_interface
        edge_interface_guess = edge_interface

        drying_finished = (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
        )
        thermal_equilibrium_reached = (
            len(center_product_history) > 10
            and abs(center_product_history[-1] - shelf_temperature) < 3.0
            and abs(edge_product_history[-1] - shelf_temperature) < 2.0
        )
        if drying_finished and thermal_equilibrium_reached:
            break

    if not records:
        raise RuntimeError("No simulation points were accepted.")

    dataframe = pd.DataFrame(records)
    summary = {
        "completed": (
            center_dry_layer >= initial_cake_height
            and edge_dry_layer >= initial_cake_height
        ),
        "total_time_min": dataframe["Time (min)"].iloc[-1],
        "center_drying_percent": dataframe["Center drying (%)"].iloc[-1],
        "edge_drying_percent": dataframe["Edge drying (%)"].iloc[-1],
        "rejected_steps": rejected_steps,
        "solver_failures": solver_failures,
    }
    return dataframe, summary


def make_excel(dataframe, parameters, summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name="Simulation", index=False)
        pd.DataFrame(
            {"Parameter": list(parameters.keys()), "Value": list(parameters.values())}
        ).to_excel(writer, sheet_name="Inputs", index=False)
        pd.DataFrame(
            {"Metric": list(summary.keys()), "Value": list(summary.values())}
        ).to_excel(writer, sheet_name="Summary", index=False)
    return output.getvalue()


# ============================================================
# SIDEBAR INPUTS
# ============================================================
with st.sidebar:
    st.header("Model inputs")
    critical_temperature = st.number_input("Critical temperature (°C)", value=-10.0)
    ice_density = st.number_input("Ice density", value=1.0, min_value=0.0001)
    sublimation_enthalpy = st.number_input("Sublimation enthalpy", value=680.0)
    fill = st.number_input("Fill volume (cm³)", value=48.0, min_value=0.001)
    outer_diameter = st.number_input("Outer diameter (cm)", value=4.7, min_value=0.001)
    wall_thickness = st.number_input("Wall thickness (cm)", value=0.17, min_value=0.0)
    pressure = st.number_input("Chamber pressure", value=0.15, min_value=0.0001, format="%.4f")
    dt = st.number_input("Time step (s)", value=1.0, min_value=0.05)
    solid_fraction = st.number_input("Solid fraction C", value=0.0625, min_value=0.0, max_value=0.99, format="%.4f")
    heat_capacity = st.number_input("Heat capacity Cp", value=134.85)

    with st.expander("Advanced parameters"):
        ice_conductivity_base = st.number_input("ki base", value=0.0059, format="%.6f")
        a5 = st.number_input("Rp coefficient a5", value=44.59)
        a6 = st.number_input("Rp coefficient a6", value=1451.73)
        a7 = st.number_input("Rp coefficient a7", value=12.93)
        initial_interface_temperature = st.number_input("Initial interface guess (°C)", value=-20.0)
        maximum_shelf_temperature = st.number_input("Maximum shelf temperature (°C)", value=30.0)
        constraint_step = st.number_input("Tcrit correction step (°C)", value=0.2, min_value=0.001)
        low_temperature_ramp = st.number_input("Shelf ramp below -10 °C (°C/min)", value=0.2)
        high_temperature_ramp = st.number_input("Shelf ramp at/above -10 °C (°C/min)", value=0.1 / 3.0, format="%.5f")
        max_duration_min = st.number_input("Maximum simulated duration (min)", value=720.0, min_value=1.0)
        max_rejected_steps = st.number_input("Maximum rejected steps", value=5000, min_value=1, step=100)
        max_solver_failures = st.number_input("Maximum solver failures", value=200, min_value=1, step=10)

    run_simulation = st.button("Run simulation", type="primary", use_container_width=True)

inner_diameter = outer_diameter - 2.0 * wall_thickness
if inner_diameter <= 0:
    st.error("Outer diameter must be greater than twice the wall thickness.")
    st.stop()

product_area = np.pi * inner_diameter**2 / 4.0
vial_area = np.pi * outer_diameter**2 / 4.0
initial_cake_height = fill / product_area

parameters = {
    "critical_temperature": critical_temperature,
    "ice_density": ice_density,
    "sublimation_enthalpy": sublimation_enthalpy,
    "fill": fill,
    "outer_diameter": outer_diameter,
    "wall_thickness": wall_thickness,
    "inner_diameter": inner_diameter,
    "product_area": product_area,
    "vial_area": vial_area,
    "initial_cake_height": initial_cake_height,
    "ice_conductivity": ice_conductivity_base * 60.0,
    "pressure": pressure,
    "dt": dt,
    "solid_fraction": solid_fraction,
    "heat_capacity": heat_capacity,
    "a5": a5,
    "a6": a6,
    "a7": a7,
    "initial_interface_temperature": initial_interface_temperature,
    "maximum_shelf_temperature": maximum_shelf_temperature,
    "constraint_step": constraint_step,
    "low_temperature_ramp": low_temperature_ramp,
    "high_temperature_ramp": high_temperature_ramp,
    "max_duration_min": max_duration_min,
    "max_rejected_steps": int(max_rejected_steps),
    "max_solver_failures": int(max_solver_failures),
}

geometry_columns = st.columns(4)
geometry_columns[0].metric("Inner diameter", f"{inner_diameter:.3f} cm")
geometry_columns[1].metric("Product area", f"{product_area:.3f} cm²")
geometry_columns[2].metric("Vial area", f"{vial_area:.3f} cm²")
geometry_columns[3].metric("Initial cake height", f"{initial_cake_height:.3f} cm")

if "simulation_data" not in st.session_state:
    st.info("Set the inputs in the sidebar and select **Run simulation**.")

if run_simulation:
    with st.spinner("Running transient simulation..."):
        try:
            simulation_data, simulation_summary = simulate(parameters)
            st.session_state["simulation_data"] = simulation_data
            st.session_state["simulation_summary"] = simulation_summary
            st.session_state["simulation_parameters"] = parameters
        except Exception as exc:
            st.exception(exc)

if "simulation_data" in st.session_state:
    df = st.session_state["simulation_data"]
    summary = st.session_state["simulation_summary"]
    used_parameters = st.session_state["simulation_parameters"]

    st.subheader("Simulation summary")
    summary_columns = st.columns(5)
    summary_columns[0].metric("Cycle time", f"{summary['total_time_min']:.2f} min")
    summary_columns[1].metric("Center drying", f"{summary['center_drying_percent']:.2f}%")
    summary_columns[2].metric("Edge drying", f"{summary['edge_drying_percent']:.2f}%")
    summary_columns[3].metric("Rejected steps", f"{summary['rejected_steps']}")
    summary_columns[4].metric("Solver failures", f"{summary['solver_failures']}")

    if summary["completed"]:
        st.success("Center and edge vial drying reached 100% within the simulated duration.")
    else:
        st.warning("Maximum simulated duration was reached before both vial positions completed drying.")

    tab_dashboard, tab_temperature, tab_cake, tab_drying, tab_rate, tab_energy, tab_data = st.tabs(
        [
            "Combined dashboard",
            "Temperature",
            "Cake height",
            "Drying %",
            "Sublimation",
            "Energy balance",
            "Data",
        ]
    )

    with tab_dashboard:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0, 0].plot(df["Time (min)"], df["Shelf temperature (°C)"], "--", label="Shelf")
        axes[0, 0].plot(df["Time (min)"], df["Center product temperature (°C)"], label="Center product")
        axes[0, 0].plot(df["Time (min)"], df["Edge product temperature (°C)"], "--", label="Edge product")
        axes[0, 0].axhline(used_parameters["critical_temperature"], color="red", linestyle="--", label="Tcrit")
        axes[0, 0].set_title("Temperature")
        axes[0, 0].set_xlabel("Time (min)")
        axes[0, 0].set_ylabel("Temperature (°C)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df["Time (min)"], df["Center remaining cake (cm)"], label="Center")
        axes[0, 1].plot(df["Time (min)"], df["Edge remaining cake (cm)"], "--", label="Edge")
        axes[0, 1].set_title("Remaining Frozen Cake Height")
        axes[0, 1].set_xlabel("Time (min)")
        axes[0, 1].set_ylabel("Height (cm)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(df["Time (min)"], df["Center drying (%)"], label="Center")
        axes[1, 0].plot(df["Time (min)"], df["Edge drying (%)"], "--", label="Edge")
        axes[1, 0].set_title("Drying Progression")
        axes[1, 0].set_xlabel("Time (min)")
        axes[1, 0].set_ylabel("Drying (%)")
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(df["Time (min)"], df["Center sublimation rate"], label="Center")
        axes[1, 1].plot(df["Time (min)"], df["Edge sublimation rate"], "--", label="Edge")
        axes[1, 1].set_title("Sublimation Rate")
        axes[1, 1].set_xlabel("Time (min)")
        axes[1, 1].set_ylabel("Sublimation rate")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_temperature:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Time (min)"], df["Shelf temperature (°C)"], "--", label="Shelf")
        ax.plot(df["Time (min)"], df["Center product temperature (°C)"], label="Center product")
        ax.plot(df["Time (min)"], df["Edge product temperature (°C)"], "--", label="Edge product")
        ax.plot(df["Time (min)"], df["Center interface temperature (°C)"], label="Center interface")
        ax.plot(df["Time (min)"], df["Edge interface temperature (°C)"], "--", label="Edge interface")
        ax.axhline(used_parameters["critical_temperature"], color="red", linestyle="--", linewidth=2, label="Tcrit")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature Profiles")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab_cake:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Time (min)"], df["Center remaining cake (cm)"], label="Center remaining cake")
        ax.plot(df["Time (min)"], df["Edge remaining cake (cm)"], "--", label="Edge remaining cake")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Remaining frozen cake height (cm)")
        ax.set_title("Remaining Frozen Cake Height")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab_drying:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Time (min)"], df["Center drying (%)"], label="Center")
        ax.plot(df["Time (min)"], df["Edge drying (%)"], "--", label="Edge")
        ax.axhline(100, color="green", linestyle="--", label="Drying endpoint")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Drying (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Drying Percentage")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab_rate:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["Time (min)"], df["Center sublimation rate"], label="Center")
        ax.plot(df["Time (min)"], df["Edge sublimation rate"], "--", label="Edge")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Sublimation rate")
        ax.set_title("Sublimation Rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab_energy:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(df["Time (min)"], df["Center heat input"], label="Heat input")
        axes[0].plot(df["Time (min)"], df["Center conduction"], "--", label="Conduction")
        axes[0].set_title("Center Vial Energy Terms")
        axes[0].set_xlabel("Time (min)")
        axes[0].set_ylabel("Energy rate")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(df["Time (min)"], df["Edge heat input"], label="Heat input")
        axes[1].plot(df["Time (min)"], df["Edge conduction"], "--", label="Conduction")
        axes[1].set_title("Edge Vial Energy Terms")
        axes[1].set_xlabel("Time (min)")
        axes[1].set_ylabel("Energy rate")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab_data:
        st.dataframe(df, use_container_width=True, height=500)

        csv_data = df.to_csv(index=False).encode("utf-8")
        excel_data = make_excel(df, used_parameters, summary)
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download CSV",
            data=csv_data,
            file_name="primary_drying_simulation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_columns[1].download_button(
            "Download Excel",
            data=excel_data,
            file_name="primary_drying_simulation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Important: confirm the unit basis of Rp, Kv, Cp, density, enthalpy, area, "
    "and time before using the model for cycle-development decisions."
)
