from enum import Enum


class ControlType(Enum):
    MANUAL = 0
    CLIENT_AP = 1
    SERVER_AP = 2
    DRIVE_MODEL = 3


class Environment(Enum):
    VOID = -1
    HIGHWAY = 0
    RURAL = 1


class NoiseMode(Enum):
    RANDOM = 0


class WeatherType(Enum):
    ALL = 0
    CLEAR = 1
    RAIN = 2


class EventType(Enum):
    LANE_TOUCH = 0
    SIDEWALK_TOUCH = 1
    OBJECT_COLLISION = 2
    REAR_END_VEHICLE_COLLISION = 3
    FRONT_END_VEHICLE_COLLISION = 4
    HLC_IGNORE = 5
    ONCOMING_LANE_WITH_RECOVERY = 6
    ONCOMING_LANE_WITHOUT_RECOVERY = 7
    STUCK = 8
