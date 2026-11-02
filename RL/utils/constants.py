DATA_SCHEMA_VERSION = 'bladed_udp_19_si_my_v1'

REAL_POWER_VARIABLE = 'I_RealPower'
ELECTRICAL_POWER_VARIABLE = 'Electrical power'

STATE_VARIABLES = [
    'I_RotorAzimuth',
    'I_RotorSpeed',
    'I_YawError',
    'I_ElectricalTorque',
    'I_GeneratorSpeed',
    REAL_POWER_VARIABLE,
    'I_PitchPosition1',
    'I_NacelleForeAftAcceleration',
    'I_NacelleSideSideAcceleration',
]

ACTION_VARIABLES = [
    'O_PcsTorqueDemand',
    'O_PitchPositionDemand1',
]

PREVIOUS_ACTION_OBSERVATION_VARIABLES = [
    'previous_action_torque_norm',
    'previous_action_pitch_norm',
]

DISTURBANCE_VARIABLES = ['I_AverageWindSpeed']

RAW_OBSERVATION_VARIABLES = [*STATE_VARIABLES, *DISTURBANCE_VARIABLES]

OBSERVATION_VARIABLES = list(RAW_OBSERVATION_VARIABLES)

# Keep near-constant but physically meaningful features in the state while
# avoiding numerical amplification from a sample IQR close to zero.
OBSERVATION_SCALE_FLOORS = {
    'I_RotorAzimuth': 1.0,
    'I_RotorSpeed': 0.01,
    'I_YawError': 0.001,
    'I_ElectricalTorque': 100.0,
    'I_GeneratorSpeed': 1.0,
    'I_RealPower': 10_000.0,
    'I_PitchPosition1': 0.01,
    'I_NacelleForeAftAcceleration': 0.01,
    'I_NacelleSideSideAcceleration': 0.01,
    'I_AverageWindSpeed': 0.1,
}

REWARD_VARIABLES = [
    ELECTRICAL_POWER_VARIABLE,
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
]

BLADED_UDP_AUXILIARY_VARIABLES = [
    'I_PitchPosition2',
    'I_PitchPosition3',
]

BLADED_UDP_VARIABLES = [
    'Time',
    'I_RotorAzimuth',
    'I_RotorSpeed',
    'I_YawError',
    'I_ElectricalTorque',
    'O_PcsTorqueDemand',
    'I_GeneratorSpeed',
    'I_PitchPosition1',
    'O_PitchPositionDemand1',
    *BLADED_UDP_AUXILIARY_VARIABLES,
    'I_NacelleForeAftAcceleration',
    'I_NacelleSideSideAcceleration',
    *DISTURBANCE_VARIABLES,
    ELECTRICAL_POWER_VARIABLE,
    REAL_POWER_VARIABLE,
    'Blade root 1 My (Nm)',
    'Blade root 2 My (Nm)',
    'Blade root 3 My (Nm)',
]

STATE_DIM = len(STATE_VARIABLES)
ACTION_DIM = len(ACTION_VARIABLES)
OBS_DIM = len(OBSERVATION_VARIABLES)
RAW_OBS_DIM = OBS_DIM
CONTROL_OBSERVATION_VARIABLES = [*OBSERVATION_VARIABLES, *PREVIOUS_ACTION_OBSERVATION_VARIABLES]
CONTROL_OBS_DIM = len(CONTROL_OBSERVATION_VARIABLES)
CONTROL_OBSERVATION_SCALE_FLOORS = [
    *[OBSERVATION_SCALE_FLOORS[name] for name in OBSERVATION_VARIABLES],
    0.01,
    0.01,
]
REWARD_DIM = len(REWARD_VARIABLES)
