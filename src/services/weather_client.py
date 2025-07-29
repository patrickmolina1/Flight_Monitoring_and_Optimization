import time
import requests
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv

load_dotenv()

class WeatherClient:
    """
    Client to fetch weather data from OpenWeatherMap API
    
    This gives us:
    - Wind speed and direction (crucial for route optimization)
    - Weather conditions
    - Temperature, pressure
    """
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"

        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY environment variable not set")
        
    def get_weather_by_coords(self, lat:float, lon:float) -> Optional[Dict]:
        """
        Get current weather conditions at specific coordinates
        
        Args:
            lat, lon: Coordinates
            
        Returns:
            Dict with weather info including wind speed/direction
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric" # Celsius, m/s for wind
            }

            response = requests.get(url,params=params)

            if response.status_code ==200:
                data = response.json()

                # Extract relevant info for flight optimization
                weather_info = {
                    'lat': lat,
                    'lon': lon,
                    'temperature': data['main']['temp'],
                    'pressure': data['main']['pressure'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),  # m/s
                    'wind_direction': data.get('wind', {}).get('deg', 0),  # degrees
                    'weather_condition': data['weather'][0]['main'],
                    'visibility': data.get('visibility', 10000),  # meters
                    'timestamp': data['dt']
                }
                return weather_info
            
            else:
                print (f"Weather API error: {response.status_code}")
                return None
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
        
    def get_route_weather (self, waypoints: List[tuple]) -> List[Dict]:
        """
        Get weather conditions along a flight route
        
        Args:
            waypoints: List of (lat, lon) tuples representing the route
            
        Returns:
            List of weather conditions at each waypoint
        """
        route_weather = []

        for lat,lon in waypoints:
            weather = self.get_weather_by_coords(lat, lon)
            if weather:
                route_weather.append(weather)
            
            time.sleep(0.1)  # Avoid hitting API rate limits

        return route_weather