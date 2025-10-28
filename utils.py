# utils.py
import pandas as pd
import numpy as np

DEFAULT_FUEL_PRICE_PER_L = 110.0  # INR per liter; adjust as needed
DEFAULT_AVG_SPEED_KMPH = 50.0     # used to estimate travel time if not present

def load_datasets(orders_path="orders.csv",
                  routes_path="routes_distance.csv",
                  fleet_path="vehicle_fleet.csv",
                  cost_path="cost_breakdown.csv"):
    orders = pd.read_csv(orders_path)
    routes = pd.read_csv(routes_path)
    fleet = pd.read_csv(fleet_path)
    cost = pd.read_csv(cost_path)
    return orders, routes, fleet, cost


def prepare_merged(orders, routes, cost):
    """
    Merge orders with routes and cost data. The dataset schema is based on your CSVs.
    """
    # routes has a 'Order_ID' column mapping each order -> route metrics
    merged = orders.merge(routes, on="Order_ID", how="left")
    merged = merged.merge(cost, on="Order_ID", how="left")
    # Ensure numeric columns exist and fillna with sensible defaults
    numeric_cols = [
        "Distance_KM", "Fuel_Consumption_L", "Toll_Charges_INR",
        "Traffic_Delay_Minutes", "Weather_Impact",
        "Fuel_Cost", "Labor_Cost", "Vehicle_Maintenance",
        "Insurance", "Packaging_Cost", "Technology_Platform_Fee", "Other_Overhead"
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
        else:
            merged[c] = 0.0

    # If orders have no weight/volume column, assume 1 unit per order
    if "Weight_KG" not in merged.columns and "Weight" not in merged.columns:
        merged["Order_Weight_KG"] = 1.0
    else:
        if "Weight_KG" in merged.columns:
            merged["Order_Weight_KG"] = pd.to_numeric(merged["Weight_KG"], errors="coerce").fillna(1.0)
        else:
            merged["Order_Weight_KG"] = pd.to_numeric(merged["Weight"], errors="coerce").fillna(1.0)

    return merged


def estimate_travel_time_hours(distance_km, traffic_delay_min, avg_speed_kmph=DEFAULT_AVG_SPEED_KMPH):
    """
    Estimate travel time in hours = (distance / speed) + (traffic_delay in hours)
    """
    travel_hours = 0.0
    if distance_km is None or distance_km == 0:
        travel_hours = 0.0
    else:
        travel_hours = distance_km / (avg_speed_kmph if avg_speed_kmph > 0 else DEFAULT_AVG_SPEED_KMPH)
    travel_hours += (traffic_delay_min or 0.0) / 60.0
    return travel_hours


def compute_order_base_cost(row):
    """
    Compute base cost per order using cost_breakdown columns and route toll.
    """
    # Sum of cost breakdown columns (they may be zero if absent)
    components = [
        row.get("Fuel_Cost", 0.0),
        row.get("Labor_Cost", 0.0),
        row.get("Vehicle_Maintenance", 0.0),
        row.get("Insurance", 0.0),
        row.get("Packaging_Cost", 0.0),
        row.get("Technology_Platform_Fee", 0.0),
        row.get("Other_Overhead", 0.0),
    ]
    tol = row.get("Toll_Charges_INR", 0.0) or 0.0
    return float(np.nansum(components)) + float(tol)


def score_vehicle_for_order(order_row, vehicle_row, avg_speed_kmph, fuel_price_per_l):
    """
    Compute raw cost, time, emissions for assigning this vehicle to this order.
    Returns dict with keys 'cost', 'time_h', 'emissions_kg'
    """
    dist = float(order_row.get("Distance_KM", 0.0) or 0.0)
    traffic_delay = float(order_row.get("Traffic_Delay_Minutes", 0.0) or 0.0)
    # Time
    time_h = estimate_travel_time_hours(dist, traffic_delay, avg_speed_kmph)

    # Fuel needed based on vehicle fuel efficiency (km per liter)
    vehicle_kmpl = vehicle_row.get("Fuel_Efficiency_KM_per_L", np.nan)
    if pd.isna(vehicle_kmpl) or vehicle_kmpl <= 0:
        # fallback to order's fuel consumption if present
        vehicle_kmpl = dist / (order_row.get("Fuel_Consumption_L", 1.0) if order_row.get("Fuel_Consumption_L", 0.0) > 0 else 1.0)

    fuel_needed_l = dist / float(vehicle_kmpl if vehicle_kmpl else 1.0)

    fuel_cost_est = fuel_needed_l * fuel_price_per_l
    toll = float(order_row.get("Toll_Charges_INR", 0.0) or 0.0)
    # base order cost (labor, packaging, platform fee, other overhead) from cost_breakdown
    base_cost = compute_order_base_cost(order_row)
    # total cost estimate (INR)
    total_cost = base_cost + fuel_cost_est

    # Emissions (kg): vehicle row provides CO2_Emissions_Kg_per_KM
    co2_kg_per_km = vehicle_row.get("CO2_Emissions_Kg_per_KM", np.nan)
    if pd.isna(co2_kg_per_km) or co2_kg_per_km == 0:
        # heuristics: derive from fuel (approx 2.392 kg CO2 per liter of fuel)
        co2_kg_per_l = 2.392
        emissions_kg = fuel_needed_l * co2_kg_per_l
    else:
        emissions_kg = dist * float(co2_kg_per_km)

    return {"cost": float(total_cost), "time_h": float(time_h), "emissions_kg": float(emissions_kg), "fuel_l": fuel_needed_l}


def greedy_assign_orders_to_vehicles(merged_orders, fleet, weights=(0.4, 0.3, 0.3),
                                    avg_speed_kmph=DEFAULT_AVG_SPEED_KMPH,
                                    fuel_price_per_l=DEFAULT_FUEL_PRICE_PER_L):
    """
    Greedy assignment:
      - For each order (in input order or by priority), evaluate all currently available vehicles
      - For the order, compute raw cost/time/emissions for each candidate vehicle
      - Normalize these three components across candidate vehicles (min-max) -> produce normalized 0..1 components
      - Compute weighted score = w_cost * norm_cost + w_time * norm_time + w_env * norm_em
      - Choose vehicle with minimum score, check capacity (vehicle capacity vs order weight). If capacity insufficient, skip vehicle.
      - Mark vehicle as used (we assume 1 order per vehicle here for simplicity, but code supports multiple orders by not removing vehicle immediately if capacity remains)
    Note: This is a baseline; can be extended to pack multiple orders into vehicles, route sequencing, or OR-Tools solver.
    """

    # copy inputs
    orders = merged_orders.copy().reset_index(drop=True)
    vehicles = fleet.copy().reset_index(drop=True)

    # Standardize string columns
    if "Status" in vehicles.columns:
        vehicles["Status"] = vehicles["Status"].astype(str)

    # Keep track of vehicles' remaining capacity (kg)
    if "Capacity_KG" in vehicles.columns:
        vehicles["Remaining_Capacity_KG"] = pd.to_numeric(vehicles["Capacity_KG"], errors="coerce").fillna(0.0)
    else:
        vehicles["Remaining_Capacity_KG"] = np.inf

    # Initially only vehicles with Status == 'Available' are candidates
    available_mask = vehicles["Status"].str.lower() == "available"
    vehicles["Available"] = available_mask

    # Prepare output list
    assignments = []

    # Optional: sort orders by Priority (Express first)
    if "Priority" in orders.columns:
        priority_map = {"Express": 0, "Standard": 1, "Economy": 2}
        orders["Priority_Sort"] = orders["Priority"].map(priority_map).fillna(1)
        orders = orders.sort_values(["Priority_Sort"]).reset_index(drop=True)

    # For each order, evaluate candidate vehicles and pick best
    for idx, order in orders.iterrows():
        # get candidate vehicles that are available and have enough remaining capacity
        order_weight = float(order.get("Order_Weight_KG", 1.0) or 1.0)
        candidate_mask = (vehicles["Available"] == True) & (vehicles["Remaining_Capacity_KG"] >= order_weight)
        candidates = vehicles[candidate_mask].copy()
        if candidates.empty:
            # No available vehicle fits -> mark unassigned
            assignments.append({
                "Order_ID": order.get("Order_ID"),
                "Origin": order.get("Origin"),
                "Destination": order.get("Destination"),
                "Assigned_Vehicle": None,
                "Vehicle_Type": None,
                "Distance_KM": float(order.get("Distance_KM", 0.0) or 0.0),
                "Estimated_Cost_INR": compute_order_base_cost(order),
                "Estimated_Time_H": estimate_travel_time_hours(float(order.get("Distance_KM", 0.0) or 0.0),
                                                              float(order.get("Traffic_Delay_Minutes", 0.0) or 0.0),
                                                              avg_speed_kmph),
                "Estimated_Emissions_KG": 0.0,
                "Status": "UNASSIGNED"
            })
            continue

        # For each candidate vehicle compute raw metrics
        metrics = []
        for vidx, veh in candidates.iterrows():
            m = score_vehicle_for_order(order, veh, avg_speed_kmph, fuel_price_per_l)
            metrics.append((vidx, m))

        # Build arrays for normalization
        costs = np.array([m[1]["cost"] for m in metrics], dtype=float)
        times = np.array([m[1]["time_h"] for m in metrics], dtype=float)
        ems = np.array([m[1]["emissions_kg"] for m in metrics], dtype=float)

        # Min-max normalize; if same value across candidates, normalization returns zeros
        def minmax_norm(arr):
            if len(arr) == 0:
                return arr
            mn = np.min(arr)
            mx = np.max(arr)
            if np.isclose(mx, mn):
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        ncosts = minmax_norm(costs)
        ntimes = minmax_norm(times)
        nems = minmax_norm(ems)

        w_cost, w_time, w_env = weights
        # compute weighted score per candidate
        scores = w_cost * ncosts + w_time * ntimes + w_env * nems

        # pick candidate with minimum score
        best_idx_in_metrics = int(np.argmin(scores))
        chosen_vidx, chosen_metrics = metrics[best_idx_in_metrics]
        chosen_vehicle = vehicles.loc[chosen_vidx]

        # assign order to chosen vehicle
        assigned_vehicle_id = chosen_vehicle.get("Vehicle_ID")
        assigned_vehicle_type = chosen_vehicle.get("Vehicle_Type")
        estimated_cost = float(chosen_metrics["cost"])
        estimated_time_h = float(chosen_metrics["time_h"])
        estimated_emissions_kg = float(chosen_metrics["emissions_kg"])

        # update vehicle remaining capacity (deduct order weight)
        vehicles.at[chosen_vidx, "Remaining_Capacity_KG"] = vehicles.at[chosen_vidx, "Remaining_Capacity_KG"] - order_weight

        # If vehicle capacity reaches near zero, mark unavailable
        if vehicles.at[chosen_vidx, "Remaining_Capacity_KG"] <= 0:
            vehicles.at[chosen_vidx, "Available"] = False

        # Optionally, if you want 1 order per vehicle, set Available=False immediately:
        # vehicles.at[chosen_vidx, "Available"] = False

        assignments.append({
            "Order_ID": order.get("Order_ID"),
            "Origin": order.get("Origin"),
            "Destination": order.get("Destination"),
            "Assigned_Vehicle": assigned_vehicle_id,
            "Vehicle_Type": assigned_vehicle_type,
            "Distance_KM": float(order.get("Distance_KM", 0.0) or 0.0),
            "Estimated_Cost_INR": round(estimated_cost, 2),
            "Estimated_Time_H": round(estimated_time_h, 3),
            "Estimated_Emissions_KG": round(estimated_emissions_kg, 3),
            "Status": "ASSIGNED"
        })

    assignment_df = pd.DataFrame(assignments)
    return assignment_df
