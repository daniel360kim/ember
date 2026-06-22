from dataclasses import dataclass
import yaml

@dataclass
class LocationConfig:
    x: float
    y: float
    z: float

@dataclass
class MomentInertiaConfig:
    Ixx: float
    Iyy: float
    Izz: float

@dataclass
class NoseConeConfig:
    type: str
    length: float
    radius: float
    
@dataclass
class BodyTubeConfig:
    type: str
    length: float
    radius: float
    


@dataclass
class VehicleConfig:
    vehicle_mass: float
    drag_coeff: float
    air_density: float
    motor: str
    cg: LocationConfig
    cp: LocationConfig
    mmoi: MomentInertiaConfig
    nose_cone: NoseConeConfig | None = None
    body_tube: BodyTubeConfig | None = None
    
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(vehicle_mass = data["vehicle_mass"],
                   motor = data["motor"],
                   drag_coeff = data["drag_coeff"],
                   air_density = data["air_density"],
                   cg = LocationConfig(**data["cg"]),
                   cp = LocationConfig(**data["cp"]),
                   mmoi = MomentInertiaConfig(**data["mmoi"]),
                   nose_cone = NoseConeConfig(**data["nose_cone"]) if "nose_cone" in data else None,
                   body_tube = BodyTubeConfig(**data["body_tube"]) if "body_tube" in data else None,
                   )