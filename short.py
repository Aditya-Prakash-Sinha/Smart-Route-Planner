import pandas as pd

files = {
    "orders": "orders.csv",
    "routes_distance": "routes_distance.csv",
    "vehicle_fleet": "vehicle_fleet.csv",
    "cost_breakdown": "cost_breakdown.csv"
}

for name, path in files.items():
    print(f"\n--- {name.upper()} ---")
    df = pd.read_csv(path)
    print(df.head())
    print("\nColumns:", df.columns.tolist())
