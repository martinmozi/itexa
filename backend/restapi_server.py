from fastapi import FastAPI
from pydantic import BaseModel, field_validator, Field
from typing import Dict, Any

fastApi = FastAPI(title="Water Tank Simulation API")

# Hardcoded maximum values
limits = {
    "MAX_WATER_LEVEL": 200.0,
    "MAX_HOLE_DIAMETER": 20.0,
    "MAX_TANK_WIDTH": 300.0,
}


class TankParameters(BaseModel):
    water_level: float = Field(..., description="Water level height in cm")
    hole_height: float = Field(..., description="Height of the hole from bottom in cm")
    hole_diameter: float = Field(..., description="Diameter of the hole in cm")
    tank_width: float = Field(..., description="Width of the tank in cm")

    @field_validator('water_level')
    @classmethod
    def water_level_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Water level must be positive")
        if v > limits["MAX_WATER_LEVEL"]:
            raise ValueError(f"Water level cannot exceed {limits["MAX_WATER_LEVEL"]} cm")
        return v

    @field_validator('hole_height')
    @classmethod
    def hole_height_must_be_valid(cls, v, info):
        if v <= 0:
            raise ValueError("Hole height must be positive")

        if 'water_level' in info.data and v > info.data['water_level']:
            raise ValueError("Hole height cannot be greater than water level")
        return v

    @field_validator('hole_diameter')
    @classmethod
    def hole_diameter_must_be_valid(cls, v):
        if v <= 0:
            raise ValueError("Hole diameter must be positive")
        if v > limits["MAX_HOLE_DIAMETER"]:
            raise ValueError(f"Hole diameter cannot exceed {limits["MAX_HOLE_DIAMETER"]} cm")
        return v

    @field_validator('tank_width')
    @classmethod
    def tank_width_must_be_valid(cls, v):
        if v <= 0:
            raise ValueError("Tank width must be positive")
        if v > limits["MAX_TANK_WIDTH"]:
            raise ValueError(f"Tank width cannot exceed {limits["MAX_TANK_WIDTH"]} cm")
        return v


@fastApi.post("/tank/simulate", response_model=Dict[str, Any])
async def simulate_tank(params: TankParameters):
    """
    Simulate water flow based on tank parameters

    - water_level: Height of water in the tank
    - hole_height: Height of the hole from the bottom
    - hole_diameter: Diameter of the hole
    - tank_width: Width of the tank

    Returns simulation parameters and a simplified flow rate calculation
    """
    fastApi.state.simulation.simulate(limits, params.water_level, params.hole_height, params.hole_diameter, params.tank_width)
    return {"status": "new simulation started"}
