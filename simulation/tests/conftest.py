from pathlib import Path

import pytest

from vehicle.config import VehicleConfig
from vehicle.atlas_v1 import build_vehicle

# Resolve the bundled Atlas config relative to the repo, so tests pass
# regardless of the directory pytest is invoked from.
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "vehicles" / "atlas.yaml")


@pytest.fixture
def config():
    return VehicleConfig.from_yaml(CONFIG_PATH)


@pytest.fixture
def vehicle(config):
    return build_vehicle(config)
