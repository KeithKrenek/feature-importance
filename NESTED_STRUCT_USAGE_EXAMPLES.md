# Nested Struct Support in signal_correlation_analysis.m

## Overview

The `signal_correlation_analysis.m` script now automatically handles nested struct hierarchies in the `data.external` field. This provides much greater flexibility in organizing your signal data.

## Features

1. **Automatic Flattening**: Nested structures are automatically flattened with underscore-separated names
2. **Validation**: All signals are validated for correct length and data type
3. **Error Handling**: Invalid signals are automatically removed with warnings
4. **Backwards Compatible**: Flat structures continue to work as before

## Examples

### Example 1: Flat Structure (Original Format)

```matlab
% Original flat structure - still works!
data.time = (0:0.01:100)';
data.response = randn(length(data.time), 1);
data.powerOn = true(length(data.time), 1);

% Flat external signals
data.external.signal1 = randn(length(data.time), 1);
data.external.signal2 = randn(length(data.time), 1);
data.external.signal3 = randn(length(data.time), 1);

% Run analysis
signal_correlation_analysis();
```

### Example 2: Nested Structure by Category

```matlab
% Time vector
data.time = (0:0.01:100)';
n = length(data.time);

% Response and power signals
data.response = randn(n, 1);
data.powerOn = true(n, 1);

% Nested external signals organized by category
data.external.environmental.temperature = 25 + 5*randn(n, 1);
data.external.environmental.pressure = 101.3 + 0.5*randn(n, 1);
data.external.environmental.humidity = 50 + 10*randn(n, 1);

data.external.electrical.voltage = 12 + 0.1*randn(n, 1);
data.external.electrical.current = 5 + 0.5*randn(n, 1);
data.external.electrical.power = 60 + 2*randn(n, 1);

data.external.mechanical.vibration = 0.1*randn(n, 1);
data.external.mechanical.torque = 10 + randn(n, 1);

% The script will automatically flatten these to:
% - environmental_temperature
% - environmental_pressure
% - environmental_humidity
% - electrical_voltage
% - electrical_current
% - electrical_power
% - mechanical_vibration
% - mechanical_torque

% Run analysis
signal_correlation_analysis();
```

### Example 3: Mixed Nested and Flat Structure

```matlab
data.time = (0:0.01:100)';
n = length(data.time);

data.response = randn(n, 1);
data.powerOn = true(n, 1);

% Mix of nested and flat signals
data.external.reference = randn(n, 1);  % Flat (top-level)

data.external.sensors.temp = 25 + randn(n, 1);      % Nested
data.external.sensors.pressure = 101 + randn(n, 1); % Nested

data.external.setpoint = 10*ones(n, 1); % Flat (top-level)

% The script will handle both formats:
% - reference (stays as is)
% - sensors_temp (flattened)
% - sensors_pressure (flattened)
% - setpoint (stays as is)

signal_correlation_analysis();
```

### Example 4: Deeply Nested Structure

```matlab
data.time = (0:0.01:100)';
n = length(data.time);

data.response = randn(n, 1);
data.powerOn = true(n, 1);

% Deeply nested structure (multiple levels)
data.external.building.floor1.room_a.temp = 20 + randn(n, 1);
data.external.building.floor1.room_b.temp = 21 + randn(n, 1);
data.external.building.floor2.room_a.temp = 19 + randn(n, 1);

data.external.hvac.zone1.fan_speed = 1000 + 100*randn(n, 1);
data.external.hvac.zone1.damper_pos = 50 + 10*randn(n, 1);

% The script will recursively flatten all levels:
% - building_floor1_room_a_temp
% - building_floor1_room_b_temp
% - building_floor2_room_a_temp
% - hvac_zone1_fan_speed
% - hvac_zone1_damper_pos

signal_correlation_analysis();
```

## What Gets Flattened

The script processes the following data types:

- ✅ **Numeric arrays** (double, single, int, etc.) → Kept as signals
- ✅ **Logical arrays** (boolean) → Kept as signals
- ✅ **Nested structs** → Recursively flattened
- ❌ **Cell arrays** → Skipped with warning
- ❌ **String/char arrays** → Skipped with warning
- ❌ **Multi-dimensional arrays** (non-vector) → Skipped with warning

## Validation Rules

After flattening, the script validates all signals:

1. **Must be a vector**: 1D array (row or column)
2. **Must match time vector length**: `length(signal) == length(data.time)`
3. **Must be numeric or logical**: No strings, cells, or objects

Invalid signals are automatically removed with warnings displayed.

## Benefits of Using Nested Structs

1. **Better Organization**: Group related signals logically
2. **Clearer Code**: Hierarchical structure is more readable
3. **Easier Maintenance**: Add/remove entire categories easily
4. **Name Preservation**: Original hierarchy preserved in flattened names

## Output

The flattened signal names appear in all output files and plots:

- Console output
- CSV summary tables (`feature_importance_summary_v2.csv`)
- Figure labels and titles
- MAT file results (`publication_grade_analysis_results_v2.mat`)

## Migration from Old Code

If you have existing code with flat structures, **no changes needed**! The script is fully backwards compatible.

To migrate to nested structures:

```matlab
% Old flat structure
data.external.temp_sensor_1 = ...;
data.external.temp_sensor_2 = ...;
data.external.pressure_sensor_1 = ...;
data.external.pressure_sensor_2 = ...;

% New nested structure (optional)
data.external.temperature.sensor_1 = ...;
data.external.temperature.sensor_2 = ...;
data.external.pressure.sensor_1 = ...;
data.external.pressure.sensor_2 = ...;
```

Both formats work identically!
