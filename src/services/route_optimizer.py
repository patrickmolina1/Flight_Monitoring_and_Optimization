import numpy as np
from typing import List, Tuple, Dict, Optional
from .weather_client import WeatherClient
import math

class RouteOptimizer:
    """
    Simple flight route optimizer considering wind conditions
    
    This creates alternative routes and picks the best one based on:
    - Distance (shorter is better)
    - Wind conditions (tailwind is good, headwind is bad)
    """
    
    def __init__(self):
        self.weather_client = WeatherClient()
    
    def calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points (Haversine formula)
        Returns distance in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def generate_route_options(self, start: Tuple[float, float], 
                              end: Tuple[float, float], 
                              num_routes: int = 3) -> List[List[Tuple[float, float]]]:
        """
        Generate alternative routes between start and end points
        
        Args:
            start: (lat, lon) of departure
            end: (lat, lon) of destination  
            num_routes: Number of alternative routes to generate
            
        Returns:
            List of routes, each route is a list of waypoints
        """
        start_lat, start_lon = start
        end_lat, end_lon = end
        
        routes = []
        
        # Route 1: Direct route
        direct_route = [start, end]
        routes.append(direct_route)
        
        # Route 2: Northern arc
        if num_routes > 1:
            mid_lat = (start_lat + end_lat) / 2 + 2.0  # 2 degrees north
            mid_lon = (start_lon + end_lon) / 2
            northern_route = [start, (mid_lat, mid_lon), end]
            routes.append(northern_route)
        
        # Route 3: Southern arc
        if num_routes > 2:
            mid_lat = (start_lat + end_lat) / 2 - 2.0  # 2 degrees south
            mid_lon = (start_lon + end_lon) / 2
            southern_route = [start, (mid_lat, mid_lon), end]
            routes.append(southern_route)
        
        return routes
    
    def evaluate_route(self, route: List[Tuple[float, float]]) -> Dict:
        """
        Evaluate a route based on distance and weather conditions
        
        Returns:
            Dict with route metrics (distance, weather_score, total_score)
        """
        total_distance = 0
        weather_scores = []
        
        # Calculate distance and get weather for each segment
        for i in range(len(route) - 1):
            lat1, lon1 = route[i]
            lat2, lon2 = route[i + 1]
            
            # Distance
            segment_distance = self.calculate_distance(lat1, lon1, lat2, lon2)
            total_distance += segment_distance
            
            # Weather at midpoint of segment
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            
            weather = self.weather_client.get_weather_by_coords(mid_lat, mid_lon)
            
            if weather:
                # Simple weather scoring (lower is better)
                wind_penalty = weather['wind_speed'] * 0.1  # Penalty for strong winds
                weather_score = wind_penalty
                weather_scores.append(weather_score)
        
        avg_weather_score = np.mean(weather_scores) if weather_scores else 0
        
        # Total score (lower is better)
        # Weight: 70% distance, 30% weather
        total_score = (total_distance * 0.7) + (avg_weather_score * 0.3)
        
        return {
            'distance_km': round(total_distance, 2),
            'weather_score': round(avg_weather_score, 2),
            'total_score': round(total_score, 2),
            'waypoints': route
        }
    
    def find_optimal_route(self, start: Tuple[float, float], 
                          end: Tuple[float, float]) -> Dict:
        """
        Find the optimal route between two points
        
        Returns:
            Dict with best route and all alternatives
        """
        print(f"🛫 Finding optimal route from {start} to {end}")
        
        # Generate route options
        routes = self.generate_route_options(start, end)
        
        # Evaluate each route
        evaluated_routes = []
        for i, route in enumerate(routes):
            print(f"   Evaluating route {i+1}...")
            evaluation = self.evaluate_route(route)
            evaluation['route_id'] = i + 1
            evaluated_routes.append(evaluation)
        
        # Find best route (lowest total score)
        best_route = min(evaluated_routes, key=lambda x: x['total_score'])
        
        return {
            'optimal_route': best_route,
            'all_routes': evaluated_routes
        }