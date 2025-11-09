# Robustness Features in signal_correlation_analysis.m

## Overview

The `signal_correlation_analysis.m` script includes comprehensive robustness features to handle unexpected data values, formatting issues, and edge cases. The script is designed to fail gracefully with informative error messages and automatically fix common data issues.

## Data Validation and Cleaning

### 1. Core Data Field Validation

**Validates:**
- `data.time`: Time vector validation
- `data.response`: Response signal validation
- `data.powerOn`: Power status validation
- `data.external`: External signals structure validation

**Automatic Fixes:**
- ✅ Row vectors → Column vectors (automatic conversion)
- ✅ Numeric `powerOn` (0/1) → Logical conversion
- ✅ NaN in `powerOn` → Treated as false (off)

**Error Detection:**
- ❌ Missing required fields
- ❌ Wrong data types
- ❌ Mismatched lengths between signals
- ❌ Non-monotonic time vectors
- ❌ NaN/Inf values in time vector
- ❌ Inf values in response/external signals
- ❌ Constant signals (zero variance)
- ❌ Empty structs

### 2. Time Vector Validation

```matlab
% Checks performed:
- Must be numeric
- Must be a vector (not matrix)
- Must be monotonically increasing
- Cannot contain NaN or Inf
- Minimum recommended length: 100 samples
- Warns if time steps > 60 seconds
```

**Example Error Messages:**
```
ERROR: data.time must be strictly monotonically increasing
WARNING: data.time is very short (50 samples) - results may be unreliable
WARNING: data.time has large time steps (120.0 sec) - may need different analysis parameters
```

### 3. Response Signal Validation

```matlab
% Checks performed:
- Must be numeric
- Must match time vector length
- Auto-converts row→column vectors
- Detects and reports NaN/Inf
- Errors if >50% NaN values
- Warns if >10% NaN values
- Errors if signal is constant
```

**Example Error Messages:**
```
ERROR: data.response contains 1250 Inf values - cannot proceed
WARNING: data.response contains 15.2% NaN values - may affect analysis quality
ERROR: data.response is constant (zero variance) - cannot analyze
```

### 4. Power-On Signal Validation

```matlab
% Checks performed:
- Must be logical or numeric 0/1
- Auto-converts numeric→logical
- Replaces NaN with false
- Checks for any events to analyze
```

**Example Messages:**
```
WARNING: data.powerOn was numeric - converted to logical
WARNING: data.powerOn contains 23 NaN values - treating as false (off)
ERROR: data.powerOn is always false - no power-on events to analyze
```

### 5. External Signals Validation

**Comprehensive per-signal checking:**

```matlab
% For each external signal:
✅ Type check (numeric/logical only)
✅ Dimension check (must be vector)
✅ Auto-fix: row→column conversion
✅ Length check (must match time vector)
✅ Inf detection (signal removed if present)
✅ NaN detection (signal kept with warning if <50%)
✅ Constant signal detection (warning)
```

**Example Messages:**
```
Auto-fixed 3 signal(s): temperature (row->col), pressure (row->col), voltage (row->col)
Removed 2 invalid signal(s)
Note: 4 signal(s) contain NaN values (will be handled during analysis)
Validated 15 external signals successfully
```

## Graceful Degradation

### Try-Catch Protection

All advanced analysis methods are wrapped in try-catch blocks:

```matlab
- ✅ Random Forest Analysis
- ✅ Mutual Information
- ✅ Partial Correlation
- ✅ Granger Causality
```

If any method fails, the script:
1. Issues a warning with the error message
2. Continues with remaining methods
3. Reports which methods succeeded at the end

**Example Output:**
```
Analysis Methods Summary:
------------------------
  Correlation Analysis:     Success
  Random Forest:            Success
  Mutual Information:       Success
  Partial Correlation:      Failed/Skipped
  Granger Causality:        Success
```

### Consensus Ranking Adaptation

The consensus ranking automatically adapts to which methods succeeded:

```matlab
% Only includes ranks from successful methods
% Automatically adjusts weighting based on available data
% Gracefully handles missing method results
```

## Common Data Issues and Automatic Fixes

### Issue 1: Row Vectors Instead of Column Vectors

**Problem:**
```matlab
data.time = [0 0.1 0.2 0.3 ...];  % Row vector (1 x N)
```

**Automatic Fix:**
```matlab
data.time = data.time(:);  % Converted to column vector (N x 1)
```

**Message:** `WARNING: data.time was a row vector - converted to column vector`

### Issue 2: Numeric Power-On Signal

**Problem:**
```matlab
data.powerOn = [1 1 1 0 0 1 ...];  % Numeric instead of logical
```

**Automatic Fix:**
```matlab
data.powerOn = logical(data.powerOn);
```

**Message:** `WARNING: data.powerOn was numeric - converted to logical`

### Issue 3: NaN Values in Signals

**Problem:**
```matlab
data.external.temperature = [25.1 NaN 25.3 25.2 NaN ...];
```

**Handling:**
- If <10% NaN: Keep signal, analysis handles NaNs
- If 10-50% NaN: Keep signal with strong warning
- If >50% NaN: Remove signal entirely

**Messages:**
```
WARNING: Signal "temperature" contains 5.2% NaN values - may reduce analysis quality
WARNING: Signal "humidity" contains 55.0% NaN values (too many) - will be skipped
```

### Issue 4: Inf Values

**Problem:**
```matlab
data.response = [0.5 1.2 Inf 0.3 ...];  % Division by zero?
```

**Handling:**
- In `response`: Critical error, cannot proceed
- In external signals: Signal removed, analysis continues

**Messages:**
```
ERROR: data.response contains 3 Inf values - cannot proceed
WARNING: Signal "velocity" contains 2 Inf values - will be skipped
```

### Issue 5: Mismatched Signal Lengths

**Problem:**
```matlab
data.time = zeros(1000, 1);
data.response = zeros(999, 1);  % Oops!
```

**Handling:** Critical error with clear message

**Message:**
```
ERROR: data.response length (999) does not match data.time length (1000)
```

### Issue 6: Nested Structs with Mixed Types

**Problem:**
```matlab
data.external.sensors.temp = [25.1 25.2 ...];     % Good
data.external.sensors.name = 'Thermocouple';      % Bad - string
data.external.sensors.calibration = {1, 2, 3};    % Bad - cell array
```

**Handling:**
```
WARNING: Skipping field "sensors_name" (type: char) - only numeric/logical arrays supported
WARNING: Skipping field "sensors_calibration" (type: cell) - only numeric/logical arrays supported
```

## Validation Summary Report

After validation, you get a comprehensive summary:

```
Validating core data fields...
  All validation checks passed (10000 samples, 1000.0 sec duration)

Processing external signals...
  Detected nested structures - flattening...
  Flattened 3 nested fields into 12 signals
  Validating and cleaning external signals...
  Auto-fixed 5 signal(s): temp_1 (row->col), temp_2 (row->col), ...
  Note: 2 signal(s) contain NaN values (will be handled during analysis)
  Validated 12 external signals successfully
```

## Best Practices for Data Preparation

### Recommended Data Structure

```matlab
% Good practice
data = struct();
data.time = (0:0.01:100)';              % Column vector, monotonic
data.response = randn(length(data.time), 1);  % Same length
data.powerOn = true(length(data.time), 1);    % Logical type

% Nested external signals
data.external.sensors.temperature = randn(length(data.time), 1);
data.external.sensors.pressure = randn(length(data.time), 1);
data.external.system.voltage = randn(length(data.time), 1);
```

### What to Avoid

```matlab
% Avoid these:
data.time = [0 0.1 0.2 ...];           % ❌ Row vector (but auto-fixed)
data.time = [0 0.1 0.05 0.2 ...];      % ❌ Non-monotonic (ERROR)
data.time = [0 NaN 0.2 ...];           % ❌ Contains NaN (ERROR)

data.response = zeros(999, 1);          % ❌ Wrong length (ERROR)
data.response = 'N/A';                  % ❌ Not numeric (ERROR)
data.response = ones(100, 1) * 5;       % ❌ Constant signal (ERROR)

data.external.signal1 = [1 2; 3 4];     % ❌ Matrix, not vector (ERROR)
data.external.signal2 = 'sensor_1';     % ❌ String (WARNING, skipped)
```

## Error Recovery Workflow

When validation fails, follow this workflow:

### Step 1: Read the Error Messages

All errors include:
- What field has the problem
- What the problem is
- What the expected format should be

### Step 2: Fix Critical Errors

Critical errors prevent analysis from starting:
- Fix missing fields
- Ensure proper data types
- Match signal lengths
- Remove NaN/Inf from time vector
- Ensure time is monotonic

### Step 3: Address Warnings (Optional)

Warnings allow analysis to continue but may affect quality:
- Consider fixing signals with many NaNs
- Review auto-converted vectors
- Check constant signals

### Step 4: Re-run Analysis

The script will validate again and report:
```
Validating core data fields...
  All validation checks passed (10000 samples, 1000.0 sec duration)
```

## Advanced: Handling Analysis Method Failures

If an analysis method fails:

### Diagnosis

Check the warning message:
```
WARNING: Random Forest analysis failed: Not enough observations
```

### Common Causes

1. **Not enough valid samples** after removing NaN/outliers
2. **Numerical issues** (ill-conditioned matrices)
3. **Missing toolbox** (e.g., Statistics and Machine Learning Toolbox)

### Impact

- Other methods still run
- Consensus ranking uses available methods only
- Results still valid, just fewer methods contributing

### Resolution

Most issues auto-resolve by:
- Collecting more data
- Reducing NaN values in signals
- Ensuring sufficient variance in signals

## Summary

The robustness features ensure:
- ✅ **Automatic fixes** for common formatting issues
- ✅ **Clear error messages** when data cannot be fixed
- ✅ **Graceful degradation** when methods fail
- ✅ **Informative reporting** throughout the process
- ✅ **Maximum analysis completion** even with imperfect data

You can confidently run the script knowing it will either:
1. Automatically fix minor issues and complete successfully
2. Clearly identify problems that need manual correction
3. Continue with partial results if some methods fail
