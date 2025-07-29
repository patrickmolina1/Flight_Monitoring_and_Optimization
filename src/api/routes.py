from fastapi import APIRouter
from src.api.flight_data import fetch_flight_data, fetch_optimized_route

router = APIRouter()

@router.get("/flights")
async def get_flights():
    return fetch_flight_data()

@router.get("/optimize-route")
async def optimize_route(departure: str, destination: str):
    return fetch_optimized_route(departure, destination)