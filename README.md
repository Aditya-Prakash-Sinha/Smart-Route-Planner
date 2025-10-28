🧭 Smart Route Planner
An intelligent routing system that optimizes for cost, time, and environmental impact.

📘 Overview
The Smart Route Planner is a Python-based decision support system that helps logistics and delivery companies plan efficient routes.
It uses optimization algorithms to minimize cost, time, and carbon footprint, providing a sustainable and data-driven approach to vehicle routing.

Built using Streamlit, this application provides an intuitive interface where users can upload datasets, view insights, and generate optimized route plans.

🎯 Objectives

Optimize delivery routes for minimum cost, minimum time, and reduced CO₂ emissions.
Provide clear visual insights and performance metrics for decision-making.
Demonstrate an AI-driven approach to sustainable logistics.

🧩 Features

✅ Intelligent route optimization using a Greedy VRP algorithm
✅ Real-time dashboard for order, fleet, and cost tracking
✅ Metrics on fuel efficiency, total cost, and environmental impact
✅ Interactive data views with filters and route summaries
✅ Downloadable CSV report for optimized routes
✅ Streamlit-powered modern user interface

🧠 Technology Stack
Layer	Tools Used
Frontend	Streamlit
Backend	Python (Pandas, NumPy, SciPy)
Optimization	Greedy Algorithm for Vehicle Routing
Visualization	Matplotlib, Seaborn, Plotly
Environmental Analysis	CO₂ Estimation using CarbonAI

📁 Project Structure
Smart Route Planner/
│
├── app.py                  # Main Streamlit application
├── utils.py                # Optimization and helper functions
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── orders.csv              # Orders dataset
├── routes_distance.csv     # Route distance dataset
├── vehicle_fleet.csv       # Vehicle fleet dataset
└── cost_breakdown.csv      # Cost and overhead dataset

⚙️ Installation and Setup

Clone the repository
git clone https://github.com/Aditya-Prakash-Sinha/Smart-Route-Planner
cd smart-route-planner
Install dependencies
pip install -r requirements.txt
Run the Streamlit application
streamlit run app.py

Upload datasets or use existing ones and start route optimization!

📊 Sample Output

Optimized Routes Table – Displays assigned vehicles, distance, cost, and CO₂ output.

Dashboard Metrics –

Total Orders
Total Distance
Available Vehicles
Total Cost
CO₂ Emissions
Charts (Optional Extension):
Cost vs Distance
Emissions per Vehicle
Fleet Utilization

🌍 Impact and Sustainability

The Smart Route Planner supports Sustainable Development Goals (SDGs):

SDG 9: Industry, Innovation & Infrastructure
SDG 11: Sustainable Cities & Communities
SDG 13: Climate Action

By optimizing routes, the system reduces fuel consumption, operational costs, and greenhouse gas emissions, promoting sustainable logistics operations.

📈 Future Enhancements

Integration with Google Maps API for live route data
Use of Genetic Algorithms or A* Search for dynamic optimization
Multi-depot and multi-vehicle type support
Dashboard analytics with emission forecasting

👨‍💻 Contributors

Aditya Prakash Sinha
AIML Developer & Data Analyst
Manipal University Jaipur
