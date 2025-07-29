import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from services.opensky_client import OpenSkyClient
from services.route_optimizer import RouteOptimizer
from models.anomaly_detector import Flight_Anomaly_Detector
import time
import json

# Page config
st.set_page_config(
    page_title="Flight Monitoring & Optimization Dashboard",
    page_icon="✈",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        color: #333 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #333 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #dc3545 !important;
        font-weight: 600;
        border-bottom: 2px solid #dc3545;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Flight Monitoring & Optimization Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time flight tracking with anomaly detection and route optimization</p>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("System Controls")

# Function to load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_flight_data():
    """Load flight data with fallback options for when APIs are rate-limited"""
    # Option 1: Use our new realistic data with real OpenSky positions
    try:
        df = pd.read_csv('data/flights_simple_realistic.csv')
        st.sidebar.success(f"Using {len(df)} flights with real OpenSky data + airline routes")
        
        # Show route quality stats
        high_confidence = len(df[df['route_confidence'] == 'high'])
        medium_confidence = len(df[df['route_confidence'] == 'medium'])
        st.sidebar.info(f"Routes: {high_confidence} airline-based, {medium_confidence} geographic")
        return df
    except FileNotFoundError:
        pass
    
    
    try:
        # Option 2: Try realistic simulation data (accurate routes)
        df = pd.read_csv('data/flights_realistic_accurate.csv')
        st.sidebar.info(f"Using {len(df)} realistic flights (OpenSky rate-limited)")
        st.sidebar.write("These routes are based on real airline patterns")
        return df
    except FileNotFoundError:
        pass
    
    try:
        # Option 4: Fallback to complete data with warning
        df = pd.read_csv('data/flights_complete.csv')
        st.sidebar.warning(f"Using synthetic routes for {len(df)} flights")
        st.sidebar.write("Routes may not match real flight plans")
        return df
    except FileNotFoundError:
        st.error("No flight data found. Please run data collection first.")
        return pd.DataFrame()

# Function to create Plotly map with click events
def create_plotly_flight_map(df, selected_flight_id=None):
    """Create an interactive Plotly map with flights and route display"""
    if df is None or df.empty:
        return None
    
    # Performance optimization: limit flights displayed if too many
    max_flights = 1000
    if len(df) > max_flights:
        # Always include anomalies and selected flight
        anomaly_df = df[df['anomaly'] == 1].copy()
        normal_df = df[df['anomaly'] == 0].copy()
        
        # Include selected flight if specified
        if selected_flight_id is not None and selected_flight_id in df.index:
            selected_df = df.loc[[selected_flight_id]].copy()
            # Remove from normal if it exists there
            if selected_flight_id in normal_df.index:
                normal_df = normal_df.drop(selected_flight_id)
        else:
            selected_df = pd.DataFrame()
        
        # Sample normal flights to fit within limit
        remaining_slots = max_flights - len(anomaly_df) - len(selected_df)
        if remaining_slots > 0 and len(normal_df) > remaining_slots:
            normal_df = normal_df.sample(n=remaining_slots, random_state=42)
        
        # Combine datasets
        display_df = pd.concat([anomaly_df, selected_df, normal_df], ignore_index=False)
        st.info(f"Displaying {len(display_df)} flights (including all {len(anomaly_df)} anomalies) out of {len(df)} total flights")
    else:
        display_df = df.copy()
    
    # Prepare data for plotting
    fig = go.Figure()
    
    # Separate flights by type for batch processing (better performance)
    normal_flights = []
    anomaly_flights = []
    selected_flight = None
    
    for idx, flight in display_df.iterrows():
        lat, lon = flight['latitude'], flight['longitude']
        callsign = flight.get('callsign', 'Unknown')
        altitude = flight.get('baro_altitude', 0)
        speed = flight.get('velocity', 0)
        
        # Fix anomaly detection - check for multiple possible anomaly representations
        anomaly_val = flight.get('anomaly', 0)
        is_anomaly = (anomaly_val == 1) or (anomaly_val == '1') or (anomaly_val == True)
        is_selected = (selected_flight_id == idx)
        
        # Route information
        dep_airport = flight.get('departure_airport_name', 'Unknown')
        arr_airport = flight.get('arrival_airport_name', 'Unknown')
        dep_city = flight.get('departure_city', 'Unknown')
        arr_city = flight.get('arrival_city', 'Unknown')
        route_distance = flight.get('route_distance_km', 0)
        airline = flight.get('airline', 'Unknown')
        
        # Create status indication
        if is_selected:
            status = "🟡 SELECTED"
        elif is_anomaly:
            status = "ANOMALY"
        else:
            status = "Normal"
        
        # Create hover text
        hover_text = f"""
        <b>{callsign}</b><br>
        Altitude: {altitude:.0f}m<br>
        Speed: {speed:.0f}m/s<br>
        From: {dep_city}<br>
        To: {arr_city}<br>
        Distance: {route_distance:.0f}km<br>
        Airline: {airline}<br>
        Status: {status}
        """
        
        # Collect flight data for batch processing
        flight_data = {
            'idx': idx,
            'lat': lat,
            'lon': lon,
            'callsign': callsign,
            'hover_text': hover_text,
            'flight': flight
        }
        
        if is_selected:
            selected_flight = flight_data
        elif is_anomaly:
            anomaly_flights.append(flight_data)
        else:
            normal_flights.append(flight_data)
    
    # Add normal flights (batch processing)
    if normal_flights:
        lats = [f['lat'] for f in normal_flights]
        lons = [f['lon'] for f in normal_flights]
        texts = [f['callsign'] for f in normal_flights]
        hovers = [f['hover_text'] for f in normal_flights]
        customdata = [f['idx'] for f in normal_flights]
        
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                symbol='circle',
                opacity=0.55
            ),
            text=texts,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hovers,
            customdata=customdata,
            name="Normal Flights",
            showlegend=True
        ))
    
    # Add anomaly flights (batch processing)
    if anomaly_flights:
        lats = [f['lat'] for f in anomaly_flights]
        lons = [f['lon'] for f in anomaly_flights]
        texts = [f['callsign'] for f in anomaly_flights]
        hovers = [f['hover_text'] for f in anomaly_flights]
        customdata = [f['idx'] for f in anomaly_flights]
        
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle',
                opacity=0.55
            ),
            text=texts,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hovers,
            customdata=customdata,
            name="Anomaly Flights",
            showlegend=True
        ))
    
    # Add selected flight (if any)
    if selected_flight:
        fig.add_trace(go.Scattermapbox(
            lat=[selected_flight['lat']],
            lon=[selected_flight['lon']],
            mode='markers',
            marker=dict(
                size=16,
                color='cyan',
                symbol='circle'
            ),
            text=[selected_flight['callsign']],
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=[selected_flight['hover_text']],
            customdata=[selected_flight['idx']],
            name="Selected Flight",
            showlegend=True
        ))
        
    # Add route line if a flight is selected
    if selected_flight and selected_flight_id == selected_flight['idx']:
        flight_data = selected_flight['flight']
        if (pd.notna(flight_data.get('departure_lat')) and pd.notna(flight_data.get('departure_lon')) and
            pd.notna(flight_data.get('arrival_lat')) and pd.notna(flight_data.get('arrival_lon'))):
            
            dep_lat = flight_data['departure_lat']
            dep_lon = flight_data['departure_lon']
            arr_lat = flight_data['arrival_lat']
            arr_lon = flight_data['arrival_lon']
            
            dep_city = flight_data.get('departure_city', 'Unknown')
            arr_city = flight_data.get('arrival_city', 'Unknown')
            dep_airport = flight_data.get('departure_airport_name', 'Unknown')
            arr_airport = flight_data.get('arrival_airport_name', 'Unknown')
            callsign = flight_data.get('callsign', 'Unknown')
            is_anomaly = flight_data.get('anomaly', 0) == 1
            
            # Add route line
            fig.add_trace(go.Scattermapbox(
                lat=[dep_lat, arr_lat],
                lon=[dep_lon, arr_lon],
                mode='lines',
                line=dict(width=4, color='orange' if is_anomaly else 'green'),
                name=f"Route: {callsign}",
                hovertemplate=f"Route: {dep_city} → {arr_city}<extra></extra>",
                showlegend=False
            ))
            
            # Add departure marker
            fig.add_trace(go.Scattermapbox(
                lat=[dep_lat],
                lon=[dep_lon],
                mode='markers',
                marker=dict(size=18, color='green', symbol='circle'),
                text=f"Departure: {dep_airport}",
                hovertemplate=f"<b>Departure</b><br>{dep_airport}<br>{dep_city}<extra></extra>",
                name="Departure",
                showlegend=False
            ))
            
            # Add arrival marker
            fig.add_trace(go.Scattermapbox(
                lat=[arr_lat],
                lon=[arr_lon],
                mode='markers',
                marker=dict(size=18, color='red', symbol='circle'),
                text=f"Arrival: {arr_airport}",
                hovertemplate=f"<b>Arrival</b><br>{arr_airport}<br>{arr_city}<extra></extra>",
                name="Arrival",
                showlegend=False
            ))
    
    # Update layout with performance optimizations
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",  # Use free OpenStreetMap instead of satellite
            center=dict(lat=display_df['latitude'].mean(), lon=display_df['longitude'].mean()),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text="Flight Map - Blue Circles: Normal | Red Circles: Anomalies | Cyan Circle: Selected",
            font=dict(color="white", size=14)  
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.8)",  # Dark background
            font=dict(
                color="white",  # White text on dark background
                size=12
            ),
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
    )
    
    return fig

# Main dashboard
def main():
    # Load data
    with st.spinner("Loading flight data..."):
        df = load_flight_data()
    
    if df is None or df.empty:
        st.error("No flight data available. Check your API connections.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Flights", len(df))
    
    with col2:
        anomaly_count = df['anomaly'].sum() if 'anomaly' in df.columns else 0
        st.metric("Anomalies", anomaly_count)
    
    with col3:
        avg_altitude = df['baro_altitude'].mean() if 'baro_altitude' in df.columns else 0
        st.metric("Avg Altitude", f"{avg_altitude:.0f}m")
    
    with col4:
        routes_with_data = len(df[df['departure_airport_name'].notna()]) if 'departure_airport_name' in df.columns else 0
        st.metric("Complete Routes", routes_with_data)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Live Map", "Anomaly Analysis", "Route Optimizer"])
    
    with tab1:
        st.subheader("Real-time Flight Tracking with Interactive Routes")
        
        # Instructions
        st.markdown(
            """
        **How to use the map:**

        - **Blue circles** = Normal flights  
        - **Red circles** = Anomaly flights  
        - **Cyan circle** = Selected flight (shows route)  
        - **Green circles** = Departure airports  
        - **Red circles** = Arrival airports  
        - **Green/Orange lines** = Flight routes  
        - **Select a flight** from the dropdown to highlight it and show its route
        """
        )
        
        # Initialize session state for selected flight
        if 'selected_flight' not in st.session_state:
            st.session_state.selected_flight = None
        
        # Flight selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Refresh button
            if st.sidebar.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.experimental_rerun()
        
        with col2:
            # Flight selector dropdown
            flight_options = ["None"] + [f"{row['callsign']} ({idx})" for idx, row in df.iterrows()]
            selected_option = st.selectbox(
                "Select flight to show route:",
                flight_options,
                key="flight_selector"
            )
            
            # Parse selected flight ID
            if selected_option != "None":
                flight_id = int(selected_option.split("(")[-1].split(")")[0])
                st.session_state.selected_flight = flight_id
            else:
                st.session_state.selected_flight = None
        
        # Create and display interactive Plotly map
        plotly_map = create_plotly_flight_map(df, st.session_state.selected_flight)
        
        if plotly_map:
            # Display the map
            st.plotly_chart(plotly_map, use_container_width=True)
        else:
            st.error("Could not create map visualization")
        
        # Show selected flight details
        if st.session_state.selected_flight is not None:
            selected_flight_data = df.iloc[st.session_state.selected_flight]
            
            st.subheader(f"Selected Flight: {selected_flight_data['callsign']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Altitude", f"{selected_flight_data['baro_altitude']:.0f}m")
            with col2:
                st.metric("Current Speed", f"{selected_flight_data['velocity']:.0f}m/s")
            with col3:
                st.metric("Route Distance", f"{selected_flight_data.get('route_distance_km', 0):.0f}km")
            
            # Route details
            if pd.notna(selected_flight_data.get('departure_city')):
                st.write(f"**Route:** {selected_flight_data['departure_city']} → {selected_flight_data['arrival_city']}")
                st.write(f"**Airline:** {selected_flight_data.get('airline', 'Unknown')}")
                
                if selected_flight_data.get('anomaly', 0):
                    st.error("**ANOMALY DETECTED** - This flight shows unusual patterns!")
                else:
                    st.success("**Normal Flight** - All parameters within expected ranges")
        
        # Clear selection button
        if st.button("Clear Route Selection"):
            st.session_state.selected_flight = None
            st.experimental_rerun()
        
        # Flight list with route information
        st.subheader("Flight Details with Routes")
        
        # Create a display dataframe with route information
        display_columns = ['callsign', 'latitude', 'longitude', 'baro_altitude', 'velocity', 
                          'departure_city', 'arrival_city', 'route_distance_km', 'airline', 'anomaly']
        available_columns = [col for col in display_columns if col in df.columns]
        display_df = df[available_columns].copy()
        
        # Rename columns for better readability
        column_names = {
            'callsign': 'Callsign',
            'latitude': 'Latitude', 
            'longitude': 'Longitude',
            'baro_altitude': 'Altitude (m)',
            'velocity': 'Speed (m/s)',
            'departure_city': 'From',
            'arrival_city': 'To',
            'route_distance_km': 'Distance (km)',
            'airline': 'Airline',
            'anomaly': 'Anomaly'
        }
        display_df = display_df.rename(columns={k: v for k, v in column_names.items() if k in display_df.columns})
        
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        st.subheader("Advanced Anomaly Detection Analysis")
        
        if 'anomaly' in df.columns:
            # Enhanced anomaly overview metrics
            anomaly_count = df['anomaly'].sum()
            total_flights = len(df)
            anomaly_rate = (anomaly_count / total_flights * 100) if total_flights > 0 else 0
            
            # Safety status indicator
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Anomalies", anomaly_count)
            
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            
            with col3:
                # Safety score based on anomaly severity
                if 'anomaly_score' in df.columns and anomaly_count > 0:
                    avg_anomaly_score = df[df['anomaly'] == 1]['anomaly_score'].mean()
                    safety_score = max(0, 100 + avg_anomaly_score * 100)  # Convert negative score to positive
                else:
                    # Fallback safety score based on anomaly rate
                    safety_score = max(0, 100 - (anomaly_rate * 2))  # Simple score based on anomaly percentage
                st.metric("Safety Score", f"{safety_score:.0f}/100")
            
            with col4:
                # Risk level indicator
                if anomaly_rate > 15:
                    risk_level = "🔴 HIGH"
                elif anomaly_rate > 8:
                    risk_level = "🟡 MEDIUM"
                else:
                    risk_level = "🟢 LOW"
                st.metric("Risk Level", risk_level)
            

            
            # Create tabs for different analysis views
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                "Pattern Analysis", "Risk Assessment", "Geographic Distribution"
            ])
            
            with analysis_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Enhanced altitude vs speed scatter plot with anomaly severity
                    # Check if anomaly_score exists, use anomaly as fallback
                    color_column = 'anomaly_score' if 'anomaly_score' in df.columns else 'anomaly'
                    size_values = np.where(df['anomaly'] == 1, 15, 8)
                    
                    fig = px.scatter(
                        df, 
                        x='velocity', 
                        y='baro_altitude',
                        color=color_column,
                        size=size_values,
                        color_continuous_scale='RdYlBu_r' if 'anomaly_score' in df.columns else None,
                        color_discrete_map={0: 'blue', 1: 'red'} if 'anomaly_score' not in df.columns else None,
                        title="Flight Pattern Analysis (Size = Anomaly Severity)",
                        labels={'velocity': 'Speed (m/s)', 'baro_altitude': 'Altitude (m)', 'color': 'Anomaly Score' if 'anomaly_score' in df.columns else 'Anomaly'},
                        hover_data=['callsign'] if 'callsign' in df.columns else None
                    )
                    
                    fig.update_layout(title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Vertical rate analysis
                    if 'vertical_rate' in df.columns:
                        fig_vertical = px.histogram(
                            df,
                            x='vertical_rate',
                            color='anomaly',
                            color_discrete_map={0: 'blue', 1: 'red'},
                            title="Vertical Rate Distribution",
                            labels={'vertical_rate': 'Vertical Rate (m/s)', 'count': 'Number of Flights'},
                            barmode='overlay',
                            opacity=0.7
                        )
                        
                        fig_vertical.update_layout(title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top')
                        fig_vertical.update_layout(height=400)
                        st.plotly_chart(fig_vertical, use_container_width=True)
                    else:
                        # Fallback: Speed distribution
                        fig_speed = px.histogram(
                            df,
                            x='velocity',
                            color='anomaly',
                            color_discrete_map={0: 'blue', 1: 'red'},
                            title="Speed Distribution Analysis",
                            labels={'velocity': 'Speed (m/s)', 'count': 'Number of Flights'},
                            barmode='overlay',
                            opacity=0.7
                        )
                        fig_speed.update_layout(title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top')
                        fig_speed.update_layout(height=400)
                        st.plotly_chart(fig_speed, use_container_width=True)
            
            with analysis_tab2:
                # Risk assessment matrix
                anomalies = df[df['anomaly'] == 1]
                if not anomalies.empty:
                    st.markdown("#### Risk Classification Overview")
                    
                    # Enhanced anomaly classification
                    risk_categories = {
                        'Critical': [],
                        'High': [],
                        'Medium': [],
                        'Low': []
                    }
                    
                    for _, flight in anomalies.iterrows():
                        altitude = flight.get('baro_altitude', 0)
                        speed = flight.get('velocity', 0)
                        vertical_rate = flight.get('vertical_rate', 0)
                        anomaly_score = flight.get('anomaly_score', 0)
                        
                        # Advanced risk scoring
                        risk_score = 0
                        risk_factors = []
                        
                        # Altitude risks
                        if altitude < 500:
                            risk_score += 10
                            risk_factors.append("Extremely Low Altitude")
                        elif altitude < 1000:
                            risk_score += 6
                            risk_factors.append("Low Altitude")
                        elif altitude > 15000:
                            risk_score += 4
                            risk_factors.append("High Altitude")
                        
                        # Speed risks
                        if speed < 50:
                            risk_score += 8
                            risk_factors.append("Critical Low Speed")
                        elif speed < 100:
                            risk_score += 5
                            risk_factors.append("Low Speed")
                        elif speed > 300:
                            risk_score += 6
                            risk_factors.append("Excessive Speed")
                        
                        # Vertical rate risks
                        if abs(vertical_rate) > 25:
                            risk_score += 7
                            risk_factors.append("Extreme Vertical Rate")
                        elif abs(vertical_rate) > 15:
                            risk_score += 4
                            risk_factors.append("High Vertical Rate")
                        
                        # Anomaly score impact
                        anomaly_score = flight.get('anomaly_score', 0)
                        if anomaly_score < -0.5:
                            risk_score += 5
                        
                        # Classify risk level
                        if risk_score >= 15:
                            category = 'Critical'
                        elif risk_score >= 10:
                            category = 'High'
                        elif risk_score >= 5:
                            category = 'Medium'
                        else:
                            category = 'Low'
                        
                        risk_categories[category].append({
                            'callsign': flight.get('callsign', 'Unknown'),
                            'risk_score': risk_score,
                            'risk_factors': ', '.join(risk_factors) if risk_factors else 'Pattern Anomaly',
                            'altitude': altitude,
                            'speed': speed,
                            'vertical_rate': vertical_rate,
                            'latitude': flight.get('latitude', None),
                            'longitude': flight.get('longitude', None),
                            'departure_lat': flight.get('departure_lat', None),
                            'departure_lon': flight.get('departure_lon', None),
                            'arrival_lat': flight.get('arrival_lat', None),
                            'arrival_lon': flight.get('arrival_lon', None),
                            'departure_city': flight.get('departure_city', 'Unknown'),
                            'arrival_city': flight.get('arrival_city', 'Unknown'),
                            'airline': flight.get('airline', 'Unknown'),
                            'flight_data': flight  # Store complete flight data for map generation
                        })
                    

                    
                    # Create summary metrics first
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        critical_count = len(risk_categories.get('Critical', []))
                        st.metric("🔴 Critical", critical_count)
                    with col2:
                        high_count = len(risk_categories.get('High', []))
                        st.metric("🟠 High", high_count)
                    with col3:
                        medium_count = len(risk_categories.get('Medium', []))
                        st.metric("🟡 Medium", medium_count)
                    with col4:
                        low_count = len(risk_categories.get('Low', []))
                        st.metric("🟢 Low", low_count)
                    
                    # Display each risk category with dropdown
                    for category, flights in risk_categories.items():
                        if flights:
                            # Create expandable section for each risk level
                            risk_icons = {
                                'Critical': '🔴',
                                'High': '🟠', 
                                'Medium': '🟡',
                                'Low': '🟢'
                            }
                            
                            with st.expander(f"{risk_icons[category]} **{category} Risk Flights** ({len(flights)} flights)", expanded=(category in ['Critical', 'High'])):
                                
                                # Create dropdown to select individual flights
                                flight_options = [f"{flight['callsign']} (Score: {flight['risk_score']})" for flight in sorted(flights, key=lambda x: x['risk_score'], reverse=True)]
                                
                                if flight_options:
                                    selected_flight_option = st.selectbox(
                                        f"Select {category.lower()} risk flight to analyze:",
                                        options=["Select a flight..."] + flight_options,
                                        key=f"flight_selector_{category}"
                                    )
                                    
                                    if selected_flight_option != "Select a flight...":
                                        # Find the selected flight data
                                        selected_callsign = selected_flight_option.split(" (Score:")[0]
                                        selected_flight = next((f for f in flights if f['callsign'] == selected_callsign), None)
                                        
                                        if selected_flight:
                                            # Display flight details
                                            st.markdown(f"### **{selected_flight['callsign']}** Analysis")
                                            
                                            # Flight metrics
                                            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                                            with detail_col1:
                                                st.metric("Altitude", f"{selected_flight['altitude']:.0f}m")
                                            with detail_col2:
                                                st.metric("Speed", f"{selected_flight['speed']:.0f}m/s")
                                            with detail_col3:
                                                st.metric("V-Rate", f"{selected_flight['vertical_rate']:.1f}m/s")
                                            with detail_col4:
                                                st.metric("Risk Score", selected_flight['risk_score'])
                                            
                                            # Route information
                                            if selected_flight['departure_city'] != 'Unknown' and selected_flight['arrival_city'] != 'Unknown':
                                                st.write(f"**Route:** {selected_flight['departure_city']} → {selected_flight['arrival_city']}")
                                                st.write(f"**Airline:** {selected_flight['airline']}")
                                            
                                            # Risk factors
                                            st.write(f"**Risk Factors:** {selected_flight['risk_factors']}")
                                            
                                            # Generate individual flight map
                                            if (selected_flight['latitude'] is not None and 
                                                selected_flight['longitude'] is not None):
                                                
                                                st.markdown("#### Flight Location & Route Map")
                                                
                                                # Create individual flight map
                                                fig_individual = go.Figure()
                                                
                                                # Add current position
                                                fig_individual.add_trace(go.Scattermapbox(
                                                    lat=[selected_flight['latitude']],
                                                    lon=[selected_flight['longitude']],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=20,
                                                        color='red' if category == 'Critical' else 'orange' if category == 'High' else 'yellow' if category == 'Medium' else 'green',
                                                        symbol='triangle-up'
                                                    ),
                                                    text=[f"� {selected_flight['callsign']}"],
                                                    hovertemplate=f"""
                                                    <b>{selected_flight['callsign']}</b><br>
                                                    Risk Level: {category}<br>
                                                    Altitude: {selected_flight['altitude']:.0f}m<br>
                                                    Speed: {selected_flight['speed']:.0f}m/s<br>
                                                    Risk Score: {selected_flight['risk_score']}<br>
                                                    Risk Factors: {selected_flight['risk_factors']}<extra></extra>
                                                    """,
                                                    name=f"{selected_flight['callsign']} (Current Position)",
                                                    showlegend=True
                                                ))
                                                
                                                # Add route line if departure and arrival coordinates are available
                                                if (selected_flight['departure_lat'] is not None and 
                                                    selected_flight['departure_lon'] is not None and
                                                    selected_flight['arrival_lat'] is not None and 
                                                    selected_flight['arrival_lon'] is not None):
                                                    
                                                    # Route line
                                                    fig_individual.add_trace(go.Scattermapbox(
                                                        lat=[selected_flight['departure_lat'], selected_flight['arrival_lat']],
                                                        lon=[selected_flight['departure_lon'], selected_flight['arrival_lon']],
                                                        mode='lines',
                                                        line=dict(width=3, color='black' if category in ['Critical', 'High'] else 'solid'),
                                                        name=f"Planned Route",
                                                        hovertemplate=f"Route: {selected_flight['departure_city']} → {selected_flight['arrival_city']}<extra></extra>",
                                                        showlegend=True
                                                    ))
                                                    
                                                    # Departure airport
                                                    fig_individual.add_trace(go.Scattermapbox(
                                                        lat=[selected_flight['departure_lat']],
                                                        lon=[selected_flight['departure_lon']],
                                                        mode='markers',
                                                        marker=dict(size=15, color='green', symbol='circle'),
                                                        text=[f"Departure: {selected_flight['departure_city']}"],
                                                        hovertemplate=f"<b>Departure</b><br>{selected_flight['departure_city']}<extra></extra>",
                                                        name="Departure",
                                                        showlegend=True
                                                    ))
                                                    
                                                    # Arrival airport
                                                    fig_individual.add_trace(go.Scattermapbox(
                                                        lat=[selected_flight['arrival_lat']],
                                                        lon=[selected_flight['arrival_lon']],
                                                        mode='markers',
                                                        marker=dict(size=15, color='red', symbol='circle'),
                                                        text=[f"Arrival: {selected_flight['arrival_city']}"],
                                                        hovertemplate=f"<b>Arrival</b><br>{selected_flight['arrival_city']}<extra></extra>",
                                                        name="Arrival",
                                                        showlegend=True
                                                    ))
                                                    
                                                    # Center map on route
                                                    center_lat = (selected_flight['departure_lat'] + selected_flight['arrival_lat'] + selected_flight['latitude']) / 3
                                                    center_lon = (selected_flight['departure_lon'] + selected_flight['arrival_lon'] + selected_flight['longitude']) / 3
                                                    zoom_level = 4
                                                else:
                                                    # Center on current position only
                                                    center_lat = selected_flight['latitude']
                                                    center_lon = selected_flight['longitude']
                                                    zoom_level = 6
                                                
                                                # Update map layout
                                                fig_individual.update_layout(
                                                    mapbox=dict(
                                                        style="open-street-map",
                                                        center=dict(lat=center_lat, lon=center_lon),
                                                        zoom=zoom_level
                                                    ),
                                                    height=500,
                                                    margin=dict(l=0, r=0, t=30, b=0),
                                                    title=dict(
                                                        text=f"{category} Risk Flight: {selected_flight['callsign']}",
                                                        font=dict(color="red" if category == 'Critical' else "orange", size=16)
                                                    ),
                                                    showlegend=True,
                                                    legend=dict(
                                                        yanchor="top",
                                                        y=0.99,
                                                        xanchor="left",
                                                        x=0.01,
                                                        bgcolor="rgba(0,0,0,0.8)",
                                                        font=dict(color="white", size=10),
                                                        bordercolor="rgba(255,255,255,0.3)",
                                                        borderwidth=1
                                                    )
                                                )
                                                
                                                st.plotly_chart(fig_individual, use_container_width=True)
                                                
                                                # Additional safety analysis
                                                st.markdown("#### Safety Analysis")
                                                
                                                safety_analysis = []
                                                if selected_flight['altitude'] < 500:
                                                    safety_analysis.append("**CRITICAL**: Extremely low altitude - Immediate attention required")
                                                elif selected_flight['altitude'] < 1000:
                                                    safety_analysis.append("**WARNING**: Low altitude - Monitor closely")
                                                
                                                if selected_flight['speed'] < 50:
                                                    safety_analysis.append("**CRITICAL**: Dangerously low speed - Potential stall risk")
                                                elif selected_flight['speed'] < 100:
                                                    safety_analysis.append("**WARNING**: Low speed - Monitor performance")
                                                elif selected_flight['speed'] > 300:
                                                    safety_analysis.append("**WARNING**: High speed - Check aircraft limits")
                                                
                                                if abs(selected_flight['vertical_rate']) > 25:
                                                    safety_analysis.append("**CRITICAL**: Extreme vertical rate - Emergency procedure check")
                                                elif abs(selected_flight['vertical_rate']) > 15:
                                                    safety_analysis.append("**WARNING**: High vertical rate - Monitor climb/descent")
                                                
                                                if safety_analysis:
                                                    for analysis in safety_analysis:
                                                        if "CRITICAL" in analysis:
                                                            st.error(analysis)
                                                        else:
                                                            st.warning(analysis)
                                                else:
                                                    st.info("ℹ️ No immediate safety concerns beyond anomaly pattern")
                                            
                                            else:
                                                st.warning("⚠️ Location data not available for map generation")
                                else:
                                    st.info("No flights in this category or flight data not found")
                else:
                    st.success("No high-risk anomalies detected!")
            
            with analysis_tab3:
                # Geographic distribution of anomalies
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    st.markdown("#### Geographic Anomaly Distribution")
                    
                    # Create density map of anomalies
                    anomaly_map_df = df[df['anomaly'] == 1].copy()
                    
                    if not anomaly_map_df.empty:
                        # Check if anomaly_score exists for size calculation
                        if 'anomaly_score' in anomaly_map_df.columns:
                            size_values = np.abs(anomaly_map_df['anomaly_score']) * 10 + 10
                            color_values = anomaly_map_df['anomaly_score']
                        else:
                            size_values = [15] * len(anomaly_map_df)  # Fixed size if no score
                            color_values = [1] * len(anomaly_map_df)  # Fixed color if no score
                        
                        fig_map = px.scatter_mapbox(
                            anomaly_map_df,
                            lat='latitude',
                            lon='longitude',
                            size=size_values,
                            color=color_values,
                            hover_data=['callsign', 'baro_altitude', 'velocity'],
                            title="Anomaly Hotspots",
                            mapbox_style="open-street-map",
                            height=500,
                            color_continuous_scale='Reds'
                        )
                        fig_map.update_layout(
                            mapbox=dict(
                                center=dict(
                                    lat=anomaly_map_df['latitude'].mean(),
                                    lon=anomaly_map_df['longitude'].mean()
                                ),
                                zoom=4
                            )
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Anomaly statistics by region
                        st.markdown("#### Regional Anomaly Statistics")
                        
                        # Simple regional grouping (can be enhanced with actual regions)
                        anomaly_map_df['lat_region'] = pd.cut(anomaly_map_df['latitude'], bins=5, labels=['South', 'South-Central', 'Central', 'North-Central', 'North'])
                        anomaly_map_df['lon_region'] = pd.cut(anomaly_map_df['longitude'], bins=5, labels=['West', 'West-Central', 'Central', 'East-Central', 'East'])
                        
                        region_stats = anomaly_map_df.groupby(['lat_region', 'lon_region']).size().reset_index(name='anomaly_count')
                        region_stats = region_stats.sort_values('anomaly_count', ascending=False).head(10)
                        
                        if not region_stats.empty:
                            st.dataframe(region_stats, use_container_width=True)
                    else:
                        st.info("No anomalies to display on map.")
            
            # Detailed anomaly table (enhanced)
            st.markdown("---")
            st.subheader("Detailed Anomaly Report")
            anomaly_columns = ['callsign', 'latitude', 'longitude', 'baro_altitude', 'velocity', 
                             'departure_city', 'arrival_city']
            # Only add anomaly_score if it exists
            if 'anomaly_score' in df.columns:
                anomaly_columns.append('anomaly_score')
            
            available_anomaly_cols = [col for col in anomaly_columns if col in df.columns]
            anomaly_details = df[df['anomaly'] == 1][available_anomaly_cols]
            
            if not anomaly_details.empty:
                # Add risk assessment to the table
                anomaly_details = anomaly_details.copy()
                
                # Calculate risk levels for each anomaly using the same logic as Risk Assessment
                risk_levels = []
                for _, flight in anomaly_details.iterrows():
                    altitude = flight.get('baro_altitude', 0)
                    speed = flight.get('velocity', 0)
                    vertical_rate = flight.get('vertical_rate', 0)
                    anomaly_score = flight.get('anomaly_score', 0)
                    
                    # Use the same advanced risk scoring as in Risk Assessment tab
                    risk_score = 0
                    
                    # Altitude risks
                    if altitude < 500:
                        risk_score += 10
                    elif altitude < 1000:
                        risk_score += 6
                    elif altitude > 15000:
                        risk_score += 4
                    
                    # Speed risks
                    if speed < 50:
                        risk_score += 8
                    elif speed < 100:
                        risk_score += 5
                    elif speed > 300:
                        risk_score += 6
                    
                    # Vertical rate risks
                    if abs(vertical_rate) > 25:
                        risk_score += 7
                    elif abs(vertical_rate) > 15:
                        risk_score += 4
                    
                    # Anomaly score impact
                    if anomaly_score < -0.5:
                        risk_score += 5
                    
                    # Classify risk level using the same thresholds as Risk Assessment
                    if risk_score >= 15:
                        risk_levels.append("🔴 Critical")
                    elif risk_score >= 10:
                        risk_levels.append("🟠 High")
                    elif risk_score >= 5:
                        risk_levels.append("🟡 Medium")
                    else:
                        risk_levels.append("🟢 Low")
                
                anomaly_details['Risk Level'] = risk_levels
                
                # Rename columns for better display
                anomaly_names = {
                    'callsign': 'Callsign',
                    'latitude': 'Latitude',
                    'longitude': 'Longitude', 
                    'baro_altitude': 'Altitude (m)',
                    'velocity': 'Speed (m/s)',
                    'departure_city': 'From',
                    'arrival_city': 'To',
                    'anomaly_score': 'Anomaly Score'
                }
                anomaly_details = anomaly_details.rename(columns={k: v for k, v in anomaly_names.items() if k in anomaly_details.columns})
                
                # Sort by risk level and anomaly score (if available)
                risk_order = {"🔴 Critical": 4, "🟠 High": 3, "🟡 Medium": 2, "🟢 Low": 1}
                anomaly_details['risk_sort'] = anomaly_details['Risk Level'].map(risk_order)
                
                if 'Anomaly Score' in anomaly_details.columns:
                    anomaly_details = anomaly_details.sort_values(['risk_sort', 'Anomaly Score'], ascending=[False, True])
                else:
                    anomaly_details = anomaly_details.sort_values('risk_sort', ascending=False)
                
                anomaly_details = anomaly_details.drop('risk_sort', axis=1)
                
                st.dataframe(anomaly_details, use_container_width=True)
            else:
                st.info("No anomalies detected in current dataset.")
                
        else:
            st.warning("Run anomaly detection first to see analysis.")
    
    with tab3:
       
        
        # Enhanced route optimization with multiple options
        st.subheader("Route Planning & Optimization")
        
        # Create tabs for different optimization modes
        route_tab1, route_tab2, route_tab3, route_tab4 = st.tabs([
            "Point-to-Point", "Multi-Route Analysis", "Weather-Aware", "Performance Comparison"
        ])
        
        with route_tab1:
            st.markdown("#### Direct Route Optimization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Departure**")
                dep_lat = st.number_input("Latitude", value=50.0379, key="dep_lat")
                dep_lon = st.number_input("Longitude", value=8.5622, key="dep_lon")
                st.caption("Default: Frankfurt Airport (FRA)")
                
                # Add preset departure airports
                departure_presets = {
                    "Frankfurt (FRA)": (50.0379, 8.5622),
                    "London Heathrow (LHR)": (51.4700, -0.4543),
                    "Paris CDG (CDG)": (49.0097, 2.5479),
                    "Amsterdam (AMS)": (52.3086, 4.7639),
                    "Munich (MUC)": (48.3538, 11.7861)
                }
                
                selected_dep = st.selectbox("Or select preset departure:", ["Custom"] + list(departure_presets.keys()))
                if selected_dep != "Custom":
                    dep_lat, dep_lon = departure_presets[selected_dep]
                    st.rerun() if hasattr(st, 'rerun') else st.experimental_rerun()
            
            with col2:
                st.write("**Destination**") 
                arr_lat = st.number_input("Latitude", value=48.3538, key="arr_lat")
                arr_lon = st.number_input("Longitude", value=14.2958, key="arr_lon")
                st.caption("Default: Linz Airport (LNZ)")
                
                # Add preset destination airports
                destination_presets = {
                    "Linz (LNZ)": (48.3538, 14.2958),
                    "Vienna (VIE)": (48.1103, 16.5697),
                    "Berlin (BER)": (52.3667, 13.5033),
                    "Rome (FCO)": (41.8003, 12.2389),
                    "Barcelona (BCN)": (41.2974, 2.0833)
                }
                
                selected_arr = st.selectbox("Or select preset destination:", ["Custom"] + list(destination_presets.keys()))
                if selected_arr != "Custom":
                    arr_lat, arr_lon = destination_presets[selected_arr]
                    st.rerun() if hasattr(st, 'rerun') else st.experimental_rerun()
            
            # Advanced optimization parameters
            st.markdown("#### ⚙️ Optimization Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fuel_priority = st.slider("Fuel Efficiency Priority", 0, 100, 40, help="Higher values prioritize fuel savings")
            
            with col2:
                time_priority = st.slider("Time Efficiency Priority", 0, 100, 35, help="Higher values prioritize shorter flight time")
            
            with col3:
                weather_priority = st.slider("Weather Avoidance Priority", 0, 100, 25, help="Higher values avoid bad weather")
            
            if st.button("Calculate Optimal Routes", type="primary"):
                with st.spinner("Analyzing multiple route options..."):
                    try:
                        optimizer = RouteOptimizer()
                        result = optimizer.find_optimal_route(
                            (dep_lat, dep_lon), 
                            (arr_lat, arr_lon)
                        )
                        
                        st.success("✅ Route analysis complete!")
                        
                        # Enhanced results display
                        best_route = result['optimal_route']
                        st.markdown("### Recommended Route")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📏 Distance", f"{best_route['distance_km']} km")
                        with col2:
                            estimated_time = best_route['distance_km'] / 800 * 60  # Rough estimate at 800 km/h
                            st.metric("⏱️ Est. Time", f"{estimated_time:.0f} min")
                        with col3:
                            st.metric("🌤️ Weather Score", f"{best_route['weather_score']}")
                        with col4:
                            efficiency = 100 - (best_route['total_score'] / 100)  # Convert to efficiency percentage
                            st.metric("⚡ Efficiency", f"{efficiency:.0f}%")
                        
                        # Route comparison table with enhanced metrics
                        st.markdown("### All Route Options")
                        
                        enhanced_routes = []
                        for route in result['all_routes']:
                            est_time = route['distance_km'] / 800 * 60
                            fuel_estimate = route['distance_km'] * 3.5  # Rough fuel estimate (kg)
                            cost_estimate = fuel_estimate * 0.8  # Rough cost estimate
                            
                            enhanced_routes.append({
                                'Route': f"Option {route['route_id']}",
                                'Distance (km)': route['distance_km'],
                                'Est. Time (min)': round(est_time, 0),
                                'Weather Score': route['weather_score'],
                                'Est. Fuel (kg)': round(fuel_estimate, 0),
                                'Est. Cost ($)': round(cost_estimate, 0),
                                'Total Score': route['total_score'],
                                'Efficiency Rank': route['route_id'] if route == best_route else ''
                            })
                        
                        route_df = pd.DataFrame(enhanced_routes)
                        
                        # Highlight best route
                        def highlight_best(row):
                            if row['Efficiency Rank'] == best_route['route_id']:
                                return ['background-color: #299438',] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(route_df.style.apply(highlight_best, axis=1), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error calculating route: {e}")
                        st.info("💡 Tip: Make sure the coordinates are valid and the weather service is available.")
        
        with route_tab2:
            st.markdown("#### Multi-Route Batch Analysis")
            
            # Allow users to analyze multiple routes from existing flight data
            if (not df.empty and 'departure_city' in df.columns and 'arrival_city' in df.columns and
                'departure_lat' in df.columns and 'departure_lon' in df.columns and
                'arrival_lat' in df.columns and 'arrival_lon' in df.columns):
                st.write("Analyze popular routes from current flight data:")
                
                # Get unique routes with valid coordinates
                routes_available = df[['departure_city', 'arrival_city', 'departure_lat', 'departure_lon', 'arrival_lat', 'arrival_lon']].dropna()
                routes_available = routes_available.drop_duplicates(subset=['departure_city', 'arrival_city'])
                
                if len(routes_available) > 0:
                    # Select routes for analysis
                    selected_routes_indices = st.multiselect(
                        "Select routes to analyze:",
                        options=range(len(routes_available)),
                        default=list(range(min(5, len(routes_available)))),  # Default to first 5
                        format_func=lambda x: f"{routes_available.iloc[x]['departure_city']} → {routes_available.iloc[x]['arrival_city']}"
                    )
                    
                    if selected_routes_indices and st.button("Analyze Selected Routes"):
                        with st.spinner("Analyzing multiple routes..."):
                            batch_results = []
                            
                            for idx in selected_routes_indices:
                                route_data = routes_available.iloc[idx]
                                try:
                                    # Validate coordinates exist and are numeric
                                    if (pd.notna(route_data['departure_lat']) and pd.notna(route_data['departure_lon']) and
                                        pd.notna(route_data['arrival_lat']) and pd.notna(route_data['arrival_lon'])):
                                        
                                        optimizer = RouteOptimizer()
                                        result = optimizer.find_optimal_route(
                                            (route_data['departure_lat'], route_data['departure_lon']),
                                            (route_data['arrival_lat'], route_data['arrival_lon'])
                                        )
                                        
                                        best_route = result['optimal_route']
                                        batch_results.append({
                                            'Route': f"{route_data['departure_city']} → {route_data['arrival_city']}",
                                            'Distance (km)': best_route['distance_km'],
                                            'Weather Score': best_route['weather_score'],
                                            'Total Score': best_route['total_score'],
                                            'Efficiency Rating': 'Excellent' if best_route['total_score'] < 500 else 'Good' if best_route['total_score'] < 1000 else 'Fair'
                                        })
                                    else:
                                        batch_results.append({
                                            'Route': f"{route_data['departure_city']} → {route_data['arrival_city']}",
                                            'Distance (km)': 'No Coords',
                                            'Weather Score': 'No Coords',
                                            'Total Score': 'No Coords',
                                            'Efficiency Rating': 'No Data'
                                        })
                                except Exception as e:
                                    batch_results.append({
                                        'Route': f"{route_data['departure_city']} → {route_data['arrival_city']}",
                                        'Distance (km)': 'Error',
                                        'Weather Score': 'Error',
                                        'Total Score': 'Error',
                                        'Efficiency Rating': 'Failed'
                                    })
                            
                            # Display results
                            if batch_results:
                                st.success(f"✅ Analyzed {len(batch_results)} routes!")
                                
                                batch_df = pd.DataFrame(batch_results)
                                st.dataframe(batch_df, use_container_width=True)
                                
                                # Visualization of batch results
                                valid_results = [r for r in batch_results if r['Distance (km)'] not in ['Error', 'No Coords']]
                                if valid_results:
                                    batch_viz_df = pd.DataFrame(valid_results)
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Distance comparison
                                        fig_distance = px.bar(
                                            batch_viz_df,
                                            x='Route',
                                            y='Distance (km)',
                                            title="Route Distance Comparison",
                                            color='Distance (km)',
                                            color_continuous_scale='Viridis'
                                        )
                                        fig_distance.update_layout(title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top')
                                        fig_distance.update_layout(xaxis_tickangle=45)
                                        st.plotly_chart(fig_distance, use_container_width=True)
                                    
                                    with col2:
                                        # Efficiency rating distribution
                                        rating_counts = batch_viz_df['Efficiency Rating'].value_counts()
                                        fig_efficiency = px.pie(
                                            values=rating_counts.values,
                                            names=rating_counts.index,
                                            title="Route Efficiency Distribution"
                                        )
                                        fig_efficiency.update_layout(title_x=0.5, title_y=0.9, title_xanchor='center', title_yanchor='top')
                                        st.plotly_chart(fig_efficiency, use_container_width=True)
                else:
                    st.info("No route data with valid coordinates available for batch analysis.")
            else:
                st.info("Load flight data with complete route information (cities and coordinates) to enable batch analysis.")
        
        with route_tab3:
            st.markdown("#### 🌤️ Weather-Aware Route Planning")
            
            
            
            # Weather impact simulation
            st.markdown("##### Weather Impact Factors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                wind_impact = st.selectbox("Wind Conditions", ["Calm", "Light Winds", "Moderate Winds", "Strong Winds", "Severe Winds"])
            
            with col2:
                precipitation = st.selectbox("Precipitation", ["Clear", "Light Rain", "Heavy Rain", "Snow", "Thunderstorms"])
            
            with col3:
                visibility = st.selectbox("Visibility", ["Excellent", "Good", "Moderate", "Poor", "Very Poor"])
            
            # Weather scoring explanation
            weather_factors = {
                "Calm": 0, "Light Winds": 2, "Moderate Winds": 5, "Strong Winds": 8, "Severe Winds": 15,
                "Clear": 0, "Light Rain": 3, "Heavy Rain": 7, "Snow": 10, "Thunderstorms": 15,
                "Excellent": 0, "Good": 1, "Moderate": 3, "Poor": 6, "Very Poor": 10
            }
            
            total_weather_impact = weather_factors[wind_impact] + weather_factors[precipitation] + weather_factors[visibility]
            
            st.markdown(f"**Current Weather Impact Score: {total_weather_impact}** (Lower is better)")
            
            if total_weather_impact <= 5:
                st.success("Excellent flying conditions")
            elif total_weather_impact <= 10:
                st.info("🟡 Good flying conditions")
            elif total_weather_impact <= 20:
                st.warning("🟠 Challenging flying conditions")
            else:
                st.error("🔴 Poor flying conditions - Consider delays")
            
            # Weather-specific route recommendations
            st.markdown("##### 📋 Weather-Based Recommendations")
            
            recommendations = []
            
            if wind_impact in ["Strong Winds", "Severe Winds"]:
                recommendations.append("🌪️ Consider wind-optimized altitude levels")
                recommendations.append("🛣️ Use routes that take advantage of jet streams")
            
            if precipitation in ["Heavy Rain", "Snow", "Thunderstorms"]:
                recommendations.append("⛈️ Plan alternate routes to avoid severe weather cells")
                recommendations.append("🛡️ Increase safety margins and fuel reserves")
            
            if visibility in ["Poor", "Very Poor"]:
                recommendations.append("👁️ Ensure IFR-certified crew and equipment")
                recommendations.append("🛬 Plan for instrument approaches at destination")
            
            if not recommendations:
                recommendations.append("✅ Standard routing procedures apply")
            
            for rec in recommendations:
                st.write(f"• {rec}")
        
        with route_tab4:
            st.markdown("#### ⚡ Performance Comparison & Analytics")
            
            # Simulate different aircraft performance for route comparison
            st.markdown("##### ✈️ Aircraft Performance Profiles")
            
            aircraft_profiles = {
                "Boeing 737-800": {"cruise_speed": 780, "fuel_consumption": 2.5, "range": 5400},
                "Airbus A320": {"cruise_speed": 800, "fuel_consumption": 2.4, "range": 6100},
                "Boeing 787": {"cruise_speed": 900, "fuel_consumption": 2.0, "range": 14800},
                "Airbus A350": {"cruise_speed": 910, "fuel_consumption": 1.9, "range": 15000},
                "Embraer E190": {"cruise_speed": 750, "fuel_consumption": 2.8, "range": 4200}
            }
            
            selected_aircraft = st.selectbox("Select Aircraft Type", list(aircraft_profiles.keys()))
            aircraft_data = aircraft_profiles[selected_aircraft]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✈️ Cruise Speed", f"{aircraft_data['cruise_speed']} km/h")
            with col2:
                st.metric("⛽ Fuel Rate", f"{aircraft_data['fuel_consumption']} kg/km")
            with col3:
                st.metric("📏 Max Range", f"{aircraft_data['range']} km")
            
            # Performance comparison for the current route
            if st.button("Analyze Aircraft Performance"):
                # Get coordinates from session state or use defaults
                try:
                    # Try to get coordinates from the Point-to-Point tab inputs
                    current_dep_lat = st.session_state.get('dep_lat', 50.0379)
                    current_dep_lon = st.session_state.get('dep_lon', 8.5622)
                    current_arr_lat = st.session_state.get('arr_lat', 48.3538)
                    current_arr_lon = st.session_state.get('arr_lon', 14.2958)
                except:
                    # Fallback to defaults if session state is not available
                    current_dep_lat, current_dep_lon = 50.0379, 8.5622  # Frankfurt
                    current_arr_lat, current_arr_lon = 48.3538, 14.2958  # Linz
                
                # Calculate basic route distance
                def calculate_distance(lat1, lon1, lat2, lon2):
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371  # Earth's radius in km
                    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    return R * c
                
                distance = calculate_distance(current_dep_lat, current_dep_lon, current_arr_lat, current_arr_lon)
                
                # Calculate performance metrics
                flight_time = distance / aircraft_data['cruise_speed'] * 60  # minutes
                fuel_needed = distance * aircraft_data['fuel_consumption']  # kg
                fuel_cost = fuel_needed * 0.8  # rough cost estimate
                
                st.markdown("##### Flight Performance Analysis")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("📏 Route Distance", f"{distance:.0f} km")
                
                with perf_col2:
                    st.metric("⏱️ Flight Time", f"{flight_time:.0f} min")
                
                with perf_col3:
                    st.metric("⛽ Fuel Required", f"{fuel_needed:.0f} kg")
                
                with perf_col4:
                    st.metric("💰 Est. Fuel Cost", f"${fuel_cost:.0f}")
                
                # Range check
                if distance > aircraft_data['range']:
                    st.error(f"⚠️ **Range Exceeded**: Distance ({distance:.0f}km) exceeds aircraft range ({aircraft_data['range']}km)")
                    st.info("💡 Consider: Fuel stops, different aircraft, or alternative routing")
                else:
                    range_utilization = (distance / aircraft_data['range']) * 100
                    st.success(f"✅ **Range OK**: Using {range_utilization:.1f}% of aircraft range")
                
                # Comparison with other aircraft
                st.markdown("##### 🔍 Aircraft Comparison")
                
                comparison_data = []
                for aircraft, specs in aircraft_profiles.items():
                    comp_flight_time = distance / specs['cruise_speed'] * 60
                    comp_fuel = distance * specs['fuel_consumption']
                    comp_cost = comp_fuel * 0.8
                    range_ok = distance <= specs['range']
                    
                    comparison_data.append({
                        'Aircraft': aircraft,
                        'Flight Time (min)': round(comp_flight_time, 0),
                        'Fuel (kg)': round(comp_fuel, 0),
                        'Cost ($)': round(comp_cost, 0),
                        'Range Suitable': '✅' if range_ok else '❌'
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)
                
                # Performance visualization
                fig_performance = px.bar(
                    comp_df,
                    x='Aircraft',
                    y=['Flight Time (min)', 'Fuel (kg)'],
                    title="Aircraft Performance Comparison",
                    barmode='group'
                )
                fig_performance.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_performance, use_container_width=True)

if __name__ == "__main__":
    main()