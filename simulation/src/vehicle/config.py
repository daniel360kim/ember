from dataclasses import dataclass
import yaml

    
@dataclass
class AeroConfig:
    air_density: float
    drag_coeff: float
    normal_force_coeff: float
    pitch_damping_coeff: float
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
class GimbalConfig:
    tau: float
    angle_limit_deg: float
    gimbal_location: LocationConfig
    
@dataclass
class MotorConfig:
    propellant_mass: float
    total_impulse: float
    total_mass: float
    motor_location: LocationConfig
    
@dataclass
class VehicleConfig:
    vehicle_mass: float
    motor: MotorConfig
    aero: AeroConfig
    cg_wet: LocationConfig
    cg_dry: LocationConfig
    cp: LocationConfig
    mmoi: MomentInertiaConfig
    gimbal_config: GimbalConfig
    nose_cone: NoseConeConfig | None = None
    body_tube: BodyTubeConfig | None = None
    
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(vehicle_mass = data["vehicle_mass_dry"],
                   motor = MotorConfig(
                       propellant_mass = data["motor"]["propellant_mass"],
                       total_impulse = data["motor"]["total_impulse"],
                       total_mass = data["motor"]["total_mass"],
                       motor_location = LocationConfig(**data["motor"]["location"])
                    ),
                   aero = AeroConfig(**data["aero"]),
                   cg_wet = LocationConfig(**data["cg_wet"]),
                   cg_dry = LocationConfig(**data["cg_dry"]),
                   cp = LocationConfig(**data["cp"]),
                   mmoi = MomentInertiaConfig(**data["mmoi"]),
                   gimbal_config = GimbalConfig(
                       tau = data["gimbal"]["tau"],
                       angle_limit_deg = data["gimbal"]["angle_limit_deg"],
                       gimbal_location = LocationConfig(**data["gimbal"]["gimbal_location"]),
                   ),
                   nose_cone = NoseConeConfig(**data["nose_cone"]) if "nose_cone" in data else None,
                   body_tube = BodyTubeConfig(**data["body_tube"]) if "body_tube" in data else None,
                   )