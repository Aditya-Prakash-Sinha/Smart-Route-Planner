# app.py
import streamlit as st
import pandas as pd
import os
from utils import load_datasets, prepare_merged, greedy_assign_orders_to_vehicles

st.set_page_config(page_title="Smart Route Planner", page_icon="🛣️", layout="wide")
st.title("🧭 Smart Route Planner — NexGen Logistics")
st.markdown("Optimize routing for **Cost**, **Time**, and **Environmental Impact**. "
            "Set weights and run the planner. The app uses your provided CSVs.")

# ------------------------
# Sidebar: Data & weights
# ------------------------
st.sidebar.header("Dataset files")
st.sidebar.write("Files loaded from current folder:")
st.sidebar.write("orders.csv, routes_distance.csv, vehicle_fleet.csv, cost_breakdown.csv")

st.sidebar.header("Objective weights (0-1)")
w_cost = st.sidebar.slider("Cost weight", 0.0, 1.0, 0.4, step=0.05)
w_time = st.sidebar.slider("Time weight", 0.0, 1.0, 0.3, step=0.05)
w_env = st.sidebar.slider("Emissions weight", 0.0, 1.0, 0.3, step=0.05)

# Normalize weights
total = w_cost + w_time + w_env
if total == 0:
    st.sidebar.error("Please set at least one non-zero weight.")
    w_cost, w_time, w_env = 0.5, 0.5, 0.0
else:
    w_cost /= total
    w_time /= total
    w_env /= total

st.sidebar.markdown(f"Normalized weights — **Cost**: {w_cost:.2f}, **Time**: {w_time:.2f}, **Emissions**: {w_env:.2f}")

st.sidebar.header("Parameters")
avg_speed = st.sidebar.number_input("Average speed (km/h) for time estimation", min_value=10, max_value=120, value=50)
fuel_price = st.sidebar.number_input("Fuel price (INR per liter)", min_value=10.0, max_value=300.0, value=110.0)

# ------------------------
# Load data
# ------------------------
@st.cache_data
def load_all():
    return load_datasets()

orders_df, routes_df, fleet_df, cost_df = load_all()

# Show top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", len(orders_df))
col2.metric("Total Routes rows", len(routes_df))
col3.metric("Total Vehicles", len(fleet_df))
col4.metric("Cost records", len(cost_df))

st.divider()

# Data previews in expanders
with st.expander("📦 Orders (sample)"):
    st.dataframe(orders_df.head())

with st.expander("🗺️ Routes (sample)"):
    st.dataframe(routes_df.head())

with st.expander("🚛 Fleet (sample)"):
    st.dataframe(fleet_df.head())

with st.expander("💰 Cost Breakdown (sample)"):
    st.dataframe(cost_df.head())

st.divider()

# ------------------------
# Run optimization
# ------------------------
st.subheader("⚙️ Run Smart Route Planner")
run = st.button("Run Planner (Greedy multi-objective)")

if run:
    with st.spinner("Preparing data and running assignment..."):
        merged = prepare_merged(orders_df, routes_df, cost_df)
        assignments = greedy_assign_orders_to_vehicles(
            merged_orders=merged,
            fleet=fleet_df,
            weights=(w_cost, w_time, w_env),
            avg_speed_kmph=avg_speed,
            fuel_price_per_l=fuel_price
        )

    st.success("Planner finished ✅")

    # Show KPIs
    total_cost = assignments["Estimated_Cost_INR"].sum()
    total_time = assignments["Estimated_Time_H"].sum()
    total_emissions = assignments["Estimated_Emissions_KG"].sum()
    assigned_count = assignments[assignments["Status"] == "ASSIGNED"].shape[0]
    unassigned_count = assignments[assignments["Status"] != "ASSIGNED"].shape[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assigned Orders", int(assigned_count))
    k2.metric("Unassigned Orders", int(unassigned_count))
    k3.metric("Total Cost (INR)", f"₹ {total_cost:,.2f}")
    k4.metric("Total Emissions (kg CO2)", f"{total_emissions:,.2f}")

    st.markdown("### 📋 Assignment snapshot")
    st.dataframe(assignments)

    # Detailed route breakdown (group by vehicle)
    st.markdown("### 🧾 Per-vehicle summary")
    if "Assigned_Vehicle" in assignments.columns:
        per_vehicle = assignments[assignments["Assigned_Vehicle"].notnull()].groupby("Assigned_Vehicle").agg({
            "Order_ID": "count",
            "Distance_KM": "sum",
            "Estimated_Cost_INR": "sum",
            "Estimated_Time_H": "sum",
            "Estimated_Emissions_KG": "sum"
        }).rename(columns={"Order_ID": "N_Orders"}).reset_index()
        st.dataframe(per_vehicle)
    else:
        st.write("No vehicle assignments produced.")

    # Export CSV
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optimized_routes.csv")
    assignments.to_csv(out_path, index=False)
    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download assignments CSV", f, file_name="optimized_routes.csv", mime="text/csv")

else:
    st.info("Set weights and click **Run Planner** to compute assignments.")
