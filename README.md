# Flight Monitoring & Optimization Dashboard

## Overview
The Flight Monitoring & Optimization Dashboard is a web application designed to monitor and optimize flight routes in real-time. It integrates live flight data and anomaly detection to provide insights into flight operations, making it a valuable tool for airlines and operational teams.

## Features
- **Live Flight Path Visualization**: Displays the current flight path on a map, allowing users to track flights in real-time.
- **Route Optimization**: Suggests optimized flight routes based on real-time weather and flight data, enhancing operational efficiency.
- **Anomaly Detection**: Monitors flight telemetry data for unusual patterns, such as altitude drops or speed spikes, and flags them for review.
- **User-Friendly Dashboard**: Provides an intuitive interface for users to interact with flight data and visualizations.

## Tech Stack
- **Backend**: Python with FastAPI for building the API and handling data integration.
- **Frontend**: React for a professional dashboard interface, utilizing Leaflet.js for map visualizations.
- **Machine Learning**: Scikit-learn for implementing anomaly detection algorithms.
- **Data Sources**: OpenSky API for real-time flight data and synthetic data generation for testing.
- **Deployment**: Options for deploying on Streamlit Cloud or AWS for a live demo.

## Setup Instructions
1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/flight-monitoring-dashboard.git
   cd flight-monitoring-dashboard
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the FastAPI server:
   ```
   uvicorn src.main:app --reload
   ```

4. **Access the Dashboard**:
   Open your web browser and navigate to `http://localhost:8000` to view the dashboard.

## Usage
- Use the dashboard to monitor live flight data and visualize flight paths.
- Explore the optimized routes suggested by the application.
- Review flagged anomalies in flight telemetry data for further analysis.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.