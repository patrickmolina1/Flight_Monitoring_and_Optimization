# Flight Monitoring & Optimization Dashboard

🚀 **Live Demo**: [https://flight-monitore-and-optimization.streamlit.app/](https://flight-monitore-and-optimization.streamlit.app/)

## Overview
The Flight Monitoring & Optimization Dashboard is a comprehensive web application designed to monitor and optimize flight routes in real-time. It integrates live flight data with advanced anomaly detection algorithms to provide actionable insights into flight operations, making it an invaluable tool for aviation professionals, analysts, and operational teams.

## ✈️ Key Features
- **🗺️ Interactive Flight Visualization**: Real-time flight tracking with interactive maps displaying current aircraft positions, flight paths, and route information
- **🛣️ Intelligent Route Optimization**: Advanced algorithms suggest optimized flight routes based on real-time weather conditions, air traffic, and operational constraints
- **🚨 Anomaly Detection**: Machine learning-powered monitoring system that identifies unusual flight patterns, altitude deviations, speed anomalies, and potential safety concerns
- **📊 Professional Dashboard**: Clean, intuitive interface with comprehensive flight metrics, risk assessments, and operational insights
- **🔍 Flight Analytics**: Detailed flight information including departure/arrival airports, airlines, aircraft types, and route confidence levels
- **⚡ Real-Time Data Processing**: Live integration with aviation APIs for up-to-date flight information and weather data

## 🛠️ Technology Stack
- **Framework**: Streamlit for rapid development and deployment of the web application
- **Backend**: Python with modular architecture for data processing and analysis
- **Data Visualization**: Plotly and Folium for interactive charts and maps
- **Machine Learning**: Scikit-learn for anomaly detection and pattern recognition algorithms
- **Data Sources**: 
  - OpenSky Network API for real-time flight tracking data
  - Airport datasets for comprehensive aviation infrastructure information
  - Weather APIs for meteorological data integration
- **Deployment**: Streamlit Cloud for free, reliable hosting with automatic updates

## 🚀 Quick Start

### Option 1: Access Live Demo
Visit the deployed application: **[https://flight-monitore-and-optimization.streamlit.app/](https://flight-monitore-and-optimization.streamlit.app/)**

### Option 2: Run Locally
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/patrickmolina1/Flight_Monitoring_and_Optimization.git
   cd Flight_Monitoring_and_Optimization
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Collect Flight Data** (Required for local setup):
   
   **Step 1**: Get raw flight data from OpenSky API
   ```bash
   cd src
   python -c "
   from services.opensky_client import OpenSkyClient
   client = OpenSkyClient()
   df = client.get_enhanced_flights_with_routes(45.0, 5.0, 55.0, 15.0)  # European region
   if df is not None:
       df.to_csv('../data/flights_raw.csv', index=False)
       print('✅ Raw flight data collected from OpenSky')
   else:
       print('❌ Failed to collect data from OpenSky API')
   "
   ```
   
   **Step 2**: Process and enhance flight data with realistic routes
   ```bash
   python simple_flight_collector.py
   ```
   
   **Step 3**: Generate complete dataset for dashboard
   ```bash
   python collect_complete_data.py
   ```
   
   **Note**: The data collection process follows this pipeline:
   - `OpenSky Client` → Raw flight positions and basic data
   - `simple_flight_collector.py` → Adds realistic routes, airline info, and airport data
   - `collect_complete_data.py` → Creates final dataset with all enhancements for dashboard

4. **Run the Dashboard**:
   ```bash
   streamlit run src/dashboard.py
   ```

5. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501` to view the dashboard.

## 📊 Dashboard Features

### Main Dashboard Sections:
- **📈 Flight Metrics Overview**: Key performance indicators and flight statistics
- **🗺️ Interactive Flight Map**: Real-time aircraft positions with detailed flight information
- **🔍 Flight Data Explorer**: Searchable and filterable flight database
- **⚠️ Anomaly Detection**: AI-powered identification of unusual flight patterns
- **🛣️ Route Optimization**: Intelligent route suggestions for operational efficiency

### Risk Assessment System:
- 🟢 **Low Risk**: Normal flight operations
- 🟡 **Medium Risk**: Minor deviations requiring monitoring
- 🟠 **High Risk**: Significant anomalies needing attention
- 🔴 **Critical Risk**: Urgent situations requiring immediate action

## 💼 Project Architecture

```
Flight_Monitoring_and_Optimization/
├── src/
│   ├── dashboard.py              # Main Streamlit dashboard application
│   ├── simple_flight_collector.py # Data collection from aviation APIs
│   ├── services/
│   │   ├── opensky_client.py     # OpenSky Network API integration
│   │   ├── airport_dataset_client.py # Airport data management
│   │   ├── weather_client.py     # Weather data integration
│   │   └── route_optimizer.py    # Route optimization algorithms
│   ├── models/
│   │   └── anomaly_detector.py   # ML-based anomaly detection
│   └── api/
│       ├── flight_data.py        # Flight data API endpoints
│       └── routes.py             # API route definitions
├── data/
│   ├── flights_simple_realistic.csv # Primary flight dataset
│   ├── airports.csv              # Airport information database
│   └── *.csv                     # Additional flight data files
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🔧 Data Collection Pipeline

The system uses a sophisticated 3-stage data collection process:

### Stage 1: Raw Data Collection (`OpenSkyClient`)
- **Source**: OpenSky Network API for live aircraft positions
- **Output**: Raw flight states (position, altitude, speed, heading)
- **Format**: Basic flight tracking data with ICAO24 identifiers

### Stage 2: Route Enhancement (`simple_flight_collector.py`)
- **Input**: Raw flight positions from Stage 1
- **Processing**: 
  - Analyzes flight callsigns to determine airlines and routes
  - Matches flights with realistic departure/arrival airports
  - Adds route confidence levels (high/medium/low)
  - Enriches with airport information and city data
- **Output**: Enhanced flight dataset with realistic route information

### Stage 3: Complete Data Processing (`collect_complete_data.py`)
- **Input**: Enhanced flight data from Stage 2
- **Processing**:
  - Applies machine learning algorithms for anomaly detection
  - Generates final dataset optimized for dashboard consumption
  - Creates backup copies and data validation
- **Output**: Production-ready flight data for dashboard visualization

### Data Flow Summary:
```
OpenSky API → Raw Positions → Route Analysis → Anomaly Detection → Dashboard
     ↓              ↓              ↓               ↓             ↓
Stage 1        Stage 2         Stage 3        Final Data    Live Demo
```

## 🎯 Use Cases

### For Aviation Professionals:
- **Air Traffic Controllers**: Monitor real-time flight positions and identify potential conflicts
- **Flight Dispatchers**: Optimize routes based on current conditions and traffic patterns
- **Safety Analysts**: Review anomaly reports and investigate unusual flight behavior
- **Operations Managers**: Track fleet performance and operational efficiency metrics

### For Researchers & Students:
- **Aviation Research**: Analyze flight patterns and operational trends
- **Data Science Projects**: Explore large-scale aviation datasets
- **Machine Learning**: Develop and test anomaly detection algorithms
- **Academic Studies**: Research air traffic management and optimization

## 🏆 Technical Highlights

- **Scalable Architecture**: Modular design supporting easy extension and maintenance
- **Real-Time Processing**: Efficient handling of live aviation data streams
- **Professional UI/UX**: Clean, intuitive interface designed for operational environments
- **Advanced Analytics**: Machine learning integration for intelligent insights
- **Data Reliability**: Robust error handling and data validation throughout the pipeline
- **Performance Optimized**: Fast loading times and responsive user interactions

## 📱 Deployment Information

**Live Application**: [https://flight-monitore-and-optimization.streamlit.app/](https://flight-monitore-and-optimization.streamlit.app/)

- **Platform**: Streamlit Cloud (Free Tier)
- **Automatic Updates**: Connected to GitHub repository for continuous deployment
- **Uptime**: 24/7 availability with automatic scaling
- **Performance**: Optimized for fast loading and responsive interactions
- **Security**: HTTPS encryption and secure data handling

## 🤝 Contributing

Contributions are welcome! This project is designed to be extensible and community-friendly.

### How to Contribute:
1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request** with a detailed description of your changes

### Contribution Areas:
- 🔧 **New Features**: Additional analytics, data sources, or visualization options
- 🐛 **Bug Fixes**: Report and fix issues you encounter
- 📖 **Documentation**: Improve README, add code comments, or create tutorials
- 🎨 **UI/UX Improvements**: Enhance the dashboard design and user experience
- ⚡ **Performance**: Optimize data processing and loading times
- 🧪 **Testing**: Add unit tests and integration tests

## 📞 Contact & Support

- **GitHub Repository**: [https://github.com/patrickmolina1/Flight_Monitoring_and_Optimization](https://github.com/patrickmolina1/Flight_Monitoring_and_Optimization)
- **Live Demo**: [https://flight-monitore-and-optimization.streamlit.app/](https://flight-monitore-and-optimization.streamlit.app/)
- **Issues & Bugs**: Please use GitHub Issues for reporting problems or requesting features
- **Developer**: Patrick Molina - Aviation Enthusiast & Software Developer

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenSky Network** for providing free access to real-time flight data
- **Streamlit** for the excellent framework and free hosting platform
- **Aviation Community** for inspiration and feedback on aviation data analysis
- **Open Source Contributors** who help improve this project

---

### 🚀 Ready to Explore Aviation Data?

Visit the live application: **[https://flight-monitore-and-optimization.streamlit.app/](https://flight-monitore-and-optimization.streamlit.app/)**

*Built with ❤️ for aviation enthusiasts and data professionals*