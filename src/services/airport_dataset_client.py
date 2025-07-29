import pandas as pd
from typing import Dict, Optional, List
import os

class AirportDatasetClient:
    """
    Client to work with local IATA/ICAO airport dataset
    """
    
    def __init__(self, dataset_path: str = 'data/airports.csv'):
        self.dataset_path = dataset_path
        self.airports_df = None
        self.airports_dict = {}
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the airport dataset"""
        try:
            if os.path.exists(self.dataset_path):
                print(f"📊 Loading airport dataset from {self.dataset_path}")
                self.airports_df = pd.read_csv(self.dataset_path)
                
                # Clean the data
                self.airports_df = self.airports_df.dropna(subset=['icao', 'latitude', 'longitude'])
                self.airports_df['icao'] = self.airports_df['icao'].str.upper().str.strip()
                self.airports_df['iata'] = self.airports_df['iata'].str.upper().str.strip()
                
                # Create lookup dictionaries for fast access
                for _, row in self.airports_df.iterrows():
                    icao = row['icao']
                    if pd.notna(icao) and icao.strip():
                        self.airports_dict[icao] = {
                            'icao_code': icao,
                            'iata_code': row.get('iata', 'Unknown'),
                            'airport_name': row.get('airport', 'Unknown'),
                            'city': row.get('region_name', 'Unknown'),
                            'country': row.get('country_code', 'Unknown'),
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'data_source': 'local_dataset'
                        }
                
                print(f"✅ Loaded {len(self.airports_dict)} airports from dataset")
                
            else:
                print(f"❌ Airport dataset not found at {self.dataset_path}")
                print("💡 Please place your airports.csv file in the data/ folder")
                
        except Exception as e:
            print(f"❌ Error loading airport dataset: {e}")
    
    def get_airport_by_icao(self, icao_code: str) -> Optional[Dict]:
        """Get airport info by ICAO code"""
        if not icao_code or not self.airports_dict:
            return None
        
        icao_code = str(icao_code).strip().upper()
        return self.airports_dict.get(icao_code)
    
    def get_airport_by_iata(self, iata_code: str) -> Optional[Dict]:
        """Get airport info by IATA code"""
        if not iata_code or self.airports_df is None:
            return None
        
        iata_code = str(iata_code).strip().upper()
        
        # Search in dataframe
        matches = self.airports_df[self.airports_df['iata'] == iata_code]
        
        if not matches.empty:
            row = matches.iloc[0]
            return {
                'icao_code': row.get('icao', 'Unknown'),
                'iata_code': row.get('iata', 'Unknown'),
                'airport_name': row.get('airport', 'Unknown'),
                'city': row.get('region_name', 'Unknown'),
                'country': row.get('country_code', 'Unknown'),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'data_source': 'local_dataset'
            }
        
        return None
    
    def get_route_coordinates(self, departure_icao: str, arrival_icao: str) -> Optional[Dict]:
        """Get complete route with coordinates"""
        departure = self.get_airport_by_icao(departure_icao)
        arrival = self.get_airport_by_icao(arrival_icao)
        
        if departure and arrival:
            return {
                'departure': departure,
                'arrival': arrival,
                'route_distance_km': self._calculate_distance(
                    departure['latitude'], departure['longitude'],
                    arrival['latitude'], arrival['longitude']
                )
            }
        
        return None
    
    def batch_lookup(self, icao_codes: List[str]) -> Dict[str, Dict]:
        """Look up multiple airports efficiently"""
        results = {}
        found = 0
        
        for icao_code in icao_codes:
            airport = self.get_airport_by_icao(icao_code)
            if airport:
                results[icao_code] = airport
                found += 1
        
        print(f"📍 Found {found} out of {len(icao_codes)} airports in dataset")
        return results
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        R = 6371
        return round(R * c, 1)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.airports_df is None:
            return {'error': 'No dataset loaded'}
        
        return {
            'total_airports': len(self.airports_dict),
            'countries': self.airports_df['country_code'].nunique(),
            'with_iata': self.airports_df['iata'].notna().sum(),
            'with_icao': self.airports_df['icao'].notna().sum(),
            'dataset_loaded': True
        }
    
    def search_airports(self, query: str, limit: int = 10) -> List[Dict]:
        """Search airports by name or code"""
        if self.airports_df is None:
            return []
        
        query = query.upper().strip()
        results = []
        
        # Search by ICAO, IATA, or airport name
        for icao, airport in self.airports_dict.items():
            if (query in icao or 
                query in airport.get('iata_code', '') or
                query.lower() in airport.get('airport_name', '').lower()):
                results.append(airport)
                
                if len(results) >= limit:
                    break
        
        return results