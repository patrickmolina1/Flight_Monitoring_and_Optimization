import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class Flight_Anomaly_Detector:
    """
    Detects anomalies in flight data using Isolation Forest
    
    What we're looking for:
    - Sudden altitude changes (emergency descents/climbs)
    - Speed anomalies (too fast/slow for altitude)
    - Unusual flight patterns
    """

    def __init__(self, contamination=0.1):
        """
        Args:
            contamination: Expected proportion of anomalies (10% is reasonable)
        """
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        self.scaler = StandardScaler()
        self.fitted = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for anomaly detection from flight data
        
        Features we'll use:
        - Altitude (baro_altitude)
        - Speed (velocity) 
        - Altitude/Speed ratio (efficiency indicator)
        - Vertical rate (climb/descent rate)
        """
        features_df = df.copy()
        
        # Clean and prepare basic features
        features_df['altitude'] = pd.to_numeric(features_df['baro_altitude'], errors='coerce')
        features_df['speed'] = pd.to_numeric(features_df['velocity'], errors='coerce')
        features_df['vertical_rate'] = pd.to_numeric(features_df['vertical_rate'], errors='coerce').fillna(0)
        
        # Remove invalid data
        features_df = features_df.dropna(subset=['altitude', 'speed'])
        features_df = features_df[features_df['altitude'] > 0]  # Remove ground level
        features_df = features_df[features_df['speed'] > 0]     # Remove stationary
        
        # Create derived features
        features_df['altitude_speed_ratio'] = features_df['altitude'] / (features_df['speed'] + 1)
        features_df['speed_per_1000ft'] = features_df['speed'] / (features_df['altitude'] / 1000 + 1)
        
        # Select features for ML
        feature_columns = ['altitude', 'speed', 'vertical_rate', 'altitude_speed_ratio', 'speed_per_1000ft']
        
        return features_df[feature_columns]
    
    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model and detect anomalies in one step
        
        Returns:
            Original dataframe with anomaly scores and labels
        """
        # Prepare features
        features_df = self.prepare_features(df)
        
        if len(features_df) < 10:
            print("Not enough valid data for anomaly detection")
            df['anomaly'] = 0
            df['anomaly_score'] = 0
            return df
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Fit and predict
        anomaly_labels = self.model.fit_predict(features_scaled)
        anomaly_scores = self.model.decision_function(features_scaled)
        
        # Add results back to original dataframe
        result_df = df.copy()
        result_df['anomaly'] = 0
        result_df['anomaly_score'] = 0
        
        # Map results back (handling index alignment)
        valid_indices = features_df.index
        result_df.loc[valid_indices, 'anomaly'] = (anomaly_labels == -1).astype(int)
        result_df.loc[valid_indices, 'anomaly_score'] = anomaly_scores
        
        self.is_fitted = True
        
        return result_df
    
    def get_anomaly_summary(self, df_with_anomalies: pd.DataFrame) -> Dict:
        """
        Generate a summary of detected anomalies
        """
        total_flights = len(df_with_anomalies)
        anomaly_count = df_with_anomalies['anomaly'].sum()
        
        if anomaly_count == 0:
            return {
                'total_flights': total_flights,
                'anomalies_detected': 0,
                'anomaly_percentage': 0,
                'anomaly_types': []
            }
        
        # Analyze anomaly types
        anomalies = df_with_anomalies[df_with_anomalies['anomaly'] == 1]
        anomaly_types = []
        
        for _, flight in anomalies.iterrows():
            altitude = flight.get('baro_altitude', 0)
            speed = flight.get('velocity', 0)
            vertical_rate = flight.get('vertical_rate', 0)
            
            anomaly_type = "Unknown"
            
            # Classify anomaly type
            if altitude < 1000:  # Very low altitude
                anomaly_type = "Low Altitude"
            elif altitude > 15000:  # Very high altitude
                anomaly_type = "High Altitude"
            elif speed > 300:  # Very high speed
                anomaly_type = "High Speed"
            elif speed < 100:  # Very low speed
                anomaly_type = "Low Speed"
            elif abs(vertical_rate) > 20:  # Rapid climb/descent
                anomaly_type = "Rapid Vertical Movement"
            
            anomaly_types.append({
                'callsign': flight.get('callsign', 'Unknown'),
                'type': anomaly_type,
                'altitude': altitude,
                'speed': speed,
                'score': flight.get('anomaly_score', 0)
            })
        
        return {
            'total_flights': total_flights,
            'anomalies_detected': anomaly_count,
            'anomaly_percentage': round((anomaly_count / total_flights) * 100, 2),
            'anomaly_types': anomaly_types
        }