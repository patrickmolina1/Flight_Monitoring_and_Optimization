import requests
import pandas as pd
from typing import List, Dict, Optional
import time

class OpenSkyClient:
    """
    Enhanced OpenSky client that gets current positions AND route data
    """
    def __init__(self):
        self.base_url = "https://opensky-network.org/api"

    def get_enhanced_flights_with_routes(self, lamin: float, lomin: float, lamax: float, lomax: float) -> Optional[pd.DataFrame]:
        """
        Get flights with current positions AND route information
        This is the main method used by CompleteFlightService
        """
        print("🔄 Getting enhanced flight data with routes...")
        
        # Step 1: Get current flight states
        df = self.get_flights_by_bbox(lamin, lomin, lamax, lomax)
        
        if df is None or df.empty:
            print("❌ No current flight data available")
            return None
        
        print(f"✅ Got {len(df)} current flight positions")
        
        # Step 2: Get route information for these flights
        icao24_list = df['icao24'].dropna().unique().tolist()
        
        # To avoid rate limiting, limit to just 3 route lookups initially
        max_route_lookups = min(3, len(icao24_list))  # Very conservative
        sample_icao24 = icao24_list[:max_route_lookups]
        
        print(f"🛫 Getting route data for {len(sample_icao24)} flights (limited to avoid rate limits)...")
        routes = self.get_flight_routes(sample_icao24)
        
        # Step 3: Merge route data with current states
        if routes:
            # Create route dataframe
            route_data = []
            for icao24, route_info in routes.items():
                route_data.append({
                    'icao24': icao24,
                    'departure_airport_icao': route_info.get('departure_airport_icao'),
                    'arrival_airport_icao': route_info.get('arrival_airport_icao'),
                    'route_callsign': route_info.get('callsign', '').strip()
                })
            
            if route_data:
                route_df = pd.DataFrame(route_data)
                
                # Merge with current states
                enhanced_df = df.merge(route_df, on='icao24', how='left')
                
                # Count successful matches
                with_routes = enhanced_df['departure_airport_icao'].notna().sum()
                print(f"✅ Enhanced data: {with_routes} flights have route information")
                
                return enhanced_df
        
        print("⚠️ No route data available, returning basic flight positions")
        
        # Add empty route columns to maintain consistency
        df['departure_airport_icao'] = None
        df['arrival_airport_icao'] = None
        df['route_callsign'] = None
        
        return df

    def get_flights_by_bbox(self, lamin: float, lomin: float, lamax: float, lomax: float) -> Optional[pd.DataFrame]:
        """Get current flight states in bounding box"""
        try:
            url = f"{self.base_url}/states/all"
            params = {
                'lamin': lamin,
                'lomin': lomin, 
                'lamax': lamax,
                'lomax': lomax
            }
            
            print(f"🗺️ Fetching current flights in region...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and 'states' in data and data['states']:
                    df = self._process_states_response(data)
                    print(f"✅ Retrieved {len(df)} flights in region")
                    return df
                else:
                    print("❌ No flights in specified region")
                    return None
            else:
                print(f"❌ OpenSky API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching flights: {e}")
            return None

    def get_flight_routes(self, icao24_list: List[str]) -> Dict[str, Dict]:
        """Get departure/arrival airports for aircraft with aggressive rate limiting"""
        routes = {}
        
        print(f"⚠️ Using conservative rate limiting due to OpenSky restrictions")
        
        for i, icao24 in enumerate(icao24_list):
            try:
                # Wait longer between requests to avoid rate limiting
                if i > 0:
                    print(f"   ⏳ Waiting 10 seconds to avoid rate limits...")
                    time.sleep(10)
                
                url = f"{self.base_url}/flights/aircraft"
                params = {
                    'icao24': icao24,
                    'begin': int(time.time()) - 24*3600,  # 1 day window
                    'end': int(time.time())  # now
                }
                
                print(f"   🔍 {icao24} ({i+1}/{len(icao24_list)})... ", end="")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and len(data) > 0:
                        # Look through all flights to find one with route data
                        found_route = False
                        for flight in data:
                            departure_airport = flight.get('estDepartureAirport')
                            arrival_airport = flight.get('estArrivalAirport')
                            
                            if departure_airport and arrival_airport:
                                routes[icao24] = {
                                    'icao24': icao24,
                                    'callsign': flight.get('callsign', '').strip(),
                                    'departure_airport_icao': departure_airport,
                                    'arrival_airport_icao': arrival_airport,
                                    'first_seen': flight.get('firstSeen'),
                                    'last_seen': flight.get('lastSeen'),
                                    'data_source': 'opensky_flights'
                                }
                                print(f"✅ {departure_airport} → {arrival_airport}")
                                found_route = True
                                break
                        
                        if not found_route:
                            print(f"❌ No route in {len(data)} flights")
                    else:
                        print("❌ No flight history")
                
                elif response.status_code == 429:
                    print("🚨 Rate limited - stopping route lookup")
                    print(f"   💡 Successfully got routes for {len(routes)} flights before limit")
                    break  # Stop immediately on rate limit
                
                else:
                    print(f"❌ HTTP {response.status_code}")
                    # Still continue with other aircraft
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n📊 Final result: {len(routes)} routes found out of {len(icao24_list)} aircraft")
        return routes

    def get_states_all(self) -> Optional[pd.DataFrame]:
        """
        Fetch all current aircraft states (for backward compatibility)
        """
        try:
            url = f"{self.base_url}/states/all"
            
            print("🌐 Fetching all flight states from OpenSky...")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and 'states' in data and data['states']:
                    df = self._process_states_response(data)
                    print(f"✅ Retrieved {len(df)} aircraft states")
                    return df
                else:
                    print("❌ No flight states available")
                    return None
            else:
                print(f"❌ OpenSky API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching flight states: {e}")
            return None

    def _process_states_response(self, data: Dict) -> pd.DataFrame:
        """Process OpenSky states response"""
        columns = [
            'icao24', 'callsign', 'origin_country', 'time_position',
            'last_contact', 'longitude', 'latitude', 'baro_altitude',
            'on_ground', 'velocity', 'true_track', 'vertical_rate',
            'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'
        ]

        df = pd.DataFrame(data['states'], columns=columns)
        
        # Clean and filter
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[df['on_ground'] == False]  # Only airborne aircraft
        
        # Clean callsigns
        df['callsign'] = df['callsign'].astype(str).str.strip()
        df['callsign'] = df['callsign'].replace('nan', None)
        
        # Convert numeric columns
        numeric_cols = ['latitude', 'longitude', 'baro_altitude', 'velocity', 'true_track']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df