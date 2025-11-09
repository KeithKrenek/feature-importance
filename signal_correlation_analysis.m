function signal_correlation_analysis()
%% ========================================================================
%  SIGNAL CORRELATION ANALYSIS
%  Analysis of External Signal Influence on Control System Response
%  ========================================================================
%
%  Features:
%  - Flexible nested struct support (automatically flattens hierarchical data)
%  - Advanced dynamic feature extraction (time constants, damping, etc.)
%  - Dynamic analysis windows (prevents event overlap)
%  - Robust feature extraction (non-zero settling, final slope)
%  - Fit-quality metrics (R-squared as a feature)
%  - Multiple statistical methods (Correlation, Random Forest, MI, Granger)
%  - Robust statistical testing with multiple comparison correction
%  - Automatic signal validation and error handling
%
%  Author: KK
%  Date: 2025
%
%% ========================================================================

clearvars; clc; close all;

%% ========================================================================
%  SECTION 1: CONFIGURATION
%  ========================================================================

fprintf('\n========================================\n');
fprintf('SIGNAL ANALYSIS\n');
fprintf('========================================\n\n');

% ---------------------- DATA LOADING ----------------------
% Replace this with your actual data loading:
% load('your_data_file.mat');
% Expected structure:
%   data.time         - Time vector [N x 1]
%   data.response     - Response signal [N x 1]
%   data.powerOn      - Boolean power status [N x 1]
%   data.external     - Struct with external signals
%
% FLEXIBILITY: data.external can be either:
%   1. Flat structure:    data.external.signal1 = [N x 1]
%                         data.external.signal2 = [N x 1]
%   2. Nested structure:  data.external.sensors.temp = [N x 1]
%                         data.external.sensors.pressure = [N x 1]
%                         data.external.actuators.voltage = [N x 1]
%
% Nested structs will be automatically flattened with underscore-separated names:
%   sensors.temp -> sensors_temp
%   sensors.pressure -> sensors_pressure
%   actuators.voltage -> actuators_voltage

data = generateSyntheticData(); % REMOVE THIS LINE for real data

% ---------------------- FLATTEN NESTED STRUCTS ----------------------
% Automatically handle nested structures in data.external
if isfield(data, 'external') && isstruct(data.external)
    fprintf('Processing external signals...\n');

    % Check if there are any nested structs
    externalFields = fieldnames(data.external);
    hasNestedStructs = false;
    for i = 1:length(externalFields)
        if isstruct(data.external.(externalFields{i}))
            hasNestedStructs = true;
            break;
        end
    end

    % Flatten if nested structs are detected
    if hasNestedStructs
        fprintf('  Detected nested structures - flattening...\n');
        originalExternal = data.external;
        data.external = flattenNestedStruct(originalExternal);

        flatFields = fieldnames(data.external);
        fprintf('  Flattened %d nested fields into %d signals\n', ...
                length(externalFields), length(flatFields));
    else
        fprintf('  External signals already in flat structure\n');
    end

    % Validate that all signals have the correct length
    fprintf('  Validating signal lengths...\n');
    expectedLength = length(data.time);
    externalFields = fieldnames(data.external);
    invalidSignals = {};

    for i = 1:length(externalFields)
        signalName = externalFields{i};
        signal = data.external.(signalName);

        % Check if signal is a vector and has correct length
        if ~isvector(signal)
            warning('Signal "%s" is not a vector (size: %s) - will be skipped', ...
                    signalName, mat2str(size(signal)));
            invalidSignals{end+1} = signalName;
        elseif length(signal) ~= expectedLength
            warning('Signal "%s" has incorrect length %d (expected %d) - will be skipped', ...
                    signalName, length(signal), expectedLength);
            invalidSignals{end+1} = signalName;
        end
    end

    % Remove invalid signals
    for i = 1:length(invalidSignals)
        data.external = rmfield(data.external, invalidSignals{i});
    end

    if ~isempty(invalidSignals)
        fprintf('  Removed %d invalid signals\n', length(invalidSignals));
    end

    finalSignalCount = length(fieldnames(data.external));
    fprintf('  Validated %d external signals successfully\n', finalSignalCount);

    if finalSignalCount == 0
        error('No valid external signals found after validation');
    end
else
    error('data.external field is missing or not a struct');
end

% ---------------------- ANALYSIS PARAMETERS ----------------------
params = struct();

% Time windows
params.preEventWindow = 5;          % seconds before power-on
params.postEventWindow = 15;        % seconds after power-on for dynamics
params.minOffDuration = 1;          % minimum power-off duration (seconds)
params.settlingThreshold = 0.02;    % 2% of initial value for settling
params.samplingRate = 1 / median(diff(data.time)); % Auto-detect sample rate
dt = 1/params.samplingRate;

% Event classification
params.cascadeWindow = 10;          % seconds - if previous event within this, flag as cascade
params.settlingCriterion = 0.05;    % 5% threshold to determine if settled before next event

% Feature extraction
params.timeLags = [0, 0.5, 1, 2, 5]; % seconds - time lags for external signals
params.expFitWindow = 5;            % seconds for exponential fit
params.frequencyBands = [0.1 1; 1 5; 5 10]; % Hz - frequency bands for spectral analysis

% Statistical methods
params.correlationMethods = {'Pearson', 'Spearman'};
params.alphaLevel = 0.05;           % significance level
params.multipleTestCorrection = 'FDR'; % 'Bonferroni', 'FDR', or 'none'
params.numBootstrap = 1000;         % bootstrap samples for CI
params.cvFolds = 5;                 % cross-validation folds

% Random Forest parameters
params.rf.numTrees = 100;
params.rf.minLeafSize = 5;

% Outlier detection
params.outlierMethod = 'MAD';       % 'MAD' (Median Absolute Deviation) or 'IQR'
params.outlierThreshold = 3;        % MAD multiplier or IQR multiplier

fprintf('Configuration loaded. Detected Sampling Rate: %.1f Hz\n\n', params.samplingRate);

%% ========================================================================
%  SECTION 2: EVENT DETECTION AND CLASSIFICATION
%  ========================================================================

fprintf('Step 1: Detecting and classifying power-on events...\n');

% Find power-on (rising) edges
powerOnEdges = diff([0; data.powerOn(:)]) > 0;
powerOnIndices = find(powerOnEdges);

%% <-- IMPROVEMENT: Find power-off (falling) edges for dynamic window
powerOffEdges = diff([1; data.powerOn(:)]) < 0; 
powerOffIndices = find(powerOffEdges);

preWindow_samples = round(params.preEventWindow / dt);
postWindow_samples = round(params.postEventWindow / dt);

% Event metadata structure
events = struct();
validEventMask = false(size(powerOnIndices));
validEventCount = 0;

for i = 1:length(powerOnIndices)
    idx = powerOnIndices(i);
    
    % Check boundaries
    if idx <= preWindow_samples || idx + postWindow_samples > length(data.time)
        continue;
    end
    
    % Find off-period start
    offStartIdx = find(diff([1; data.powerOn(1:idx)]) < 0, 1, 'last');
    if isempty(offStartIdx)
        continue;
    end
    
    % Check minimum off duration
    offDuration = (idx - offStartIdx) * dt;
    if offDuration < params.minOffDuration
        continue;
    end
    
    % Check for cascading events
    isCascade = false;
    initialCondition = data.response(idx);
    timeSincePrevious = inf;
    responseMagnitudeBeforeOff = abs(data.response(offStartIdx));
    
    if i > 1 && any(validEventMask(1:i-1))
        lastValidEventIdx = find(validEventMask(1:i-1), 1, 'last');
        timeSincePrevious = (idx - powerOnIndices(lastValidEventIdx)) * dt;
        
        if timeSincePrevious < params.cascadeWindow
            isCascade = true;
            % Check if system had settled
            prevResponseAtPowerOn = data.response(powerOnIndices(lastValidEventIdx));
            if responseMagnitudeBeforeOff > params.settlingCriterion * abs(prevResponseAtPowerOn)
                isCascade = true;
            end
        end
    end
    
    % Mark as valid and store metadata
    validEventMask(i) = true;
    validEventCount = validEventCount + 1;
    
    events(validEventCount).index = idx;
    events(validEventCount).time = data.time(idx);
    events(validEventCount).offStartIdx = offStartIdx;
    events(validEventCount).offDuration = offDuration;
    events(validEventCount).isCascade = isCascade;
    events(validEventCount).timeSincePrevious = timeSincePrevious;
    events(validEventCount).initialCondition = initialCondition;
    events(validEventCount).responseMagnitudeBeforeOff = responseMagnitudeBeforeOff;
    
    %% <-- IMPROVEMENT: Find time until the *next* power-off event
    nextOffEvent = powerOffIndices(find(powerOffIndices > idx, 1, 'first'));
    if isempty(nextOffEvent)
        events(validEventCount).timeUntilNextOff = inf;
    else
        events(validEventCount).timeUntilNextOff = (nextOffEvent - idx) * dt;
    end
    
end

numEvents = length(events);
fprintf('  Found %d valid power-on events\n', numEvents);

if numEvents > 0
    fprintf('  - Clean starts: %d\n', sum(~[events.isCascade]));
    fprintf('  - Cascade events: %d\n\n', sum([events.isCascade]));
else
    fprintf('  - No valid events found.\n\n');
end

if numEvents < 10
    warning('Low sample size (n=%d). Results may lack statistical power.', numEvents);
    if numEvents == 0
        error('No valid events found. Aborting analysis.');
    end
end

%% ========================================================================
%  SECTION 3: ADVANCED RESPONSE SIGNAL FEATURE EXTRACTION
%  ========================================================================

fprintf('Step 2: Extracting advanced response signal features...\n');

responseFeatures = struct();
responseFeatureNames = {};

responseFeatureNames{end+1} = 'valueAtPowerOn';
responseFeatures.valueAtPowerOn = nan(numEvents, 1);
responseFeatureNames{end+1} = 'initialCondition';
responseFeatures.initialCondition = nan(numEvents, 1);
responseFeatureNames{end+1} = 'absValueAtPowerOn';
responseFeatures.absValueAtPowerOn = nan(numEvents, 1);
responseFeatureNames{end+1} = 'initialSlope';
responseFeatures.initialSlope = nan(numEvents, 1);
responseFeatureNames{end+1} = 'slopeAt1s';
responseFeatures.slopeAt1s = nan(numEvents, 1);
responseFeatureNames{end+1} = 'slopeAt3s';
responseFeatures.slopeAt3s = nan(numEvents, 1);
responseFeatureNames{end+1} = 'timeConstant';
responseFeatures.timeConstant = nan(numEvents, 1);
responseFeatureNames{end+1} = 'dampingRatio';
responseFeatures.dampingRatio = nan(numEvents, 1);
responseFeatureNames{end+1} = 'naturalFrequency';
responseFeatures.naturalFrequency = nan(numEvents, 1);
responseFeatureNames{end+1} = 'riseTime90';
responseFeatures.riseTime90 = nan(numEvents, 1);
responseFeatureNames{end+1} = 'settlingTime2pct';
responseFeatures.settlingTime2pct = nan(numEvents, 1);
responseFeatureNames{end+1} = 'settlingTime5pct';
responseFeatures.settlingTime5pct = nan(numEvents, 1);
responseFeatureNames{end+1} = 'overshoot';
responseFeatures.overshoot = nan(numEvents, 1);
responseFeatureNames{end+1} = 'peakRecovery';
responseFeatures.peakRecovery = nan(numEvents, 1);
responseFeatureNames{end+1} = 'peakTime';
responseFeatures.peakTime = nan(numEvents, 1);
responseFeatureNames{end+1} = 'rmsError';
responseFeatures.rmsError = nan(numEvents, 1);
responseFeatureNames{end+1} = 'mae';
responseFeatures.mae = nan(numEvents, 1);
responseFeatureNames{end+1} = 'iae';
responseFeatures.iae = nan(numEvents, 1);
responseFeatureNames{end+1} = 'ise';
responseFeatures.ise = nan(numEvents, 1);
responseFeatureNames{end+1} = 'itae';
responseFeatures.itae = nan(numEvents, 1);
responseFeatureNames{end+1} = 'dominantFrequency';
responseFeatures.dominantFrequency = nan(numEvents, 1);
responseFeatureNames{end+1} = 'spectralPower_lowFreq';
responseFeatures.spectralPower_lowFreq = nan(numEvents, 1);
responseFeatureNames{end+1} = 'spectralPower_midFreq';
responseFeatures.spectralPower_midFreq = nan(numEvents, 1);
responseFeatureNames{end+1} = 'spectralPower_highFreq';
responseFeatures.spectralPower_highFreq = nan(numEvents, 1);
responseFeatureNames{end+1} = 'recoveryRate';
responseFeatures.recoveryRate = nan(numEvents, 1);
responseFeatureNames{end+1} = 'isCascadeEvent';
responseFeatures.isCascadeEvent = nan(numEvents, 1);
responseFeatureNames{end+1} = 'offDuration';
responseFeatures.offDuration = nan(numEvents, 1);

responseFeatureNames{end+1} = 'finalSettlingValue';
responseFeatures.finalSettlingValue = nan(numEvents, 1);
responseFeatureNames{end+1} = 'finalSlope';
responseFeatures.finalSlope = nan(numEvents, 1);
responseFeatureNames{end+1} = 'timeConstant_R2';
responseFeatures.timeConstant_R2 = nan(numEvents, 1);

for i = 1:numEvents
    idx = events(i).index;
    
    % Basic features
    responseFeatures.valueAtPowerOn(i) = events(i).initialCondition;
    responseFeatures.initialCondition(i) = events(i).initialCondition;
    responseFeatures.absValueAtPowerOn(i) = abs(events(i).initialCondition);
    responseFeatures.isCascadeEvent(i) = double(events(i).isCascade);
    responseFeatures.offDuration(i) = events(i).offDuration;
    
    %% <-- IMPROVEMENT: Use dynamic window based on timeUntilNextOff
    % Calculate the dynamic window size (max_window OR time_to_next_event)
    % Subtract 1 sample to avoid overlapping with the off-event itself
    dynamicPostSamples = min(postWindow_samples, round(events(i).timeUntilNextOff / dt) - 1);
    if dynamicPostSamples <= 1
        continue; % Not enough data to analyze this event
    end
    
    % Post power-on segment
    postSegment = data.response(idx:min(idx + dynamicPostSamples, length(data.response)));
    timePost = (0:length(postSegment)-1)' * dt;
    
    % Initial slope (first 0.5 seconds)
    slopeWindow1 = min(round(0.5/dt), length(postSegment));
    if slopeWindow1 > 2
        p = polyfit(timePost(1:slopeWindow1), postSegment(1:slopeWindow1), 1);
        responseFeatures.initialSlope(i) = p(1);
    end
    
    % Slope at 1s and 3s
    idx1s = min(round(1/dt), length(postSegment));
    if idx1s > 10
        window = max(1, idx1s-5):min(length(postSegment), idx1s+5);
        p = polyfit(timePost(window), postSegment(window), 1);
        responseFeatures.slopeAt1s(i) = p(1);
    end
    
    idx3s = min(round(3/dt), length(postSegment));
    if idx3s > 10
        window = max(1, idx3s-5):min(length(postSegment), idx3s+5);
        p = polyfit(timePost(window), postSegment(window), 1);
        responseFeatures.slopeAt3s(i) = p(1);
    end
    
    %% <-- IMPROVEMENT: Add finalSlope (drift) calculation
    % Final slope (last 2 seconds of *dynamic* window)
    finalSlopeWindowStart = max(1, length(postSegment) - round(2/dt));
    finalSlopeWindowEnd = length(postSegment);
    if (finalSlopeWindowEnd - finalSlopeWindowStart) > 10
        p = polyfit(timePost(finalSlopeWindowStart:finalSlopeWindowEnd), postSegment(finalSlopeWindowStart:finalSlopeWindowEnd), 1);
        responseFeatures.finalSlope(i) = p(1);
    end
    
    % Time constant estimation (exponential fit)
    fitWindow = min(round(params.expFitWindow/dt), length(postSegment));
    if fitWindow > 10 && abs(postSegment(1)) > 1e-6
        try
            % Fit: y = A*exp(-t/tau) + C
            fo = fitoptions('Method', 'NonlinearLeastSquares', ...
                           'Lower', [-Inf, 0, -Inf], ...
                           'Upper', [Inf, Inf, Inf], ...
                           'StartPoint', [postSegment(1), 1, mean(postSegment(end-5:end))]);
            ft = fittype('A*exp(-x/tau) + C', 'options', fo);
            [fitResult, gof] = fit(timePost(1:fitWindow), postSegment(1:fitWindow), ft);
            
            %% <-- IMPROVEMENT: Store fit quality (R2) as a feature
            responseFeatures.timeConstant_R2(i) = gof.rsquare;
            
            if gof.rsquare > 0.5
                responseFeatures.timeConstant(i) = fitResult.tau;
                %% <-- IMPROVEMENT: Store non-zero settling value (C)
                responseFeatures.finalSettlingValue(i) = fitResult.C;
            end
        catch
            % warning('Exponential fit failed for event %d', i);
        end
    end
    
    % Damping ratio and natural frequency (for oscillatory response)
    % Look for zero crossings and peaks
    zeroCrossings = find(diff(sign(postSegment)) ~= 0);
    if length(zeroCrossings) >= 2
        % Estimate natural frequency from zero crossings
        period = 2 * mean(diff(timePost(zeroCrossings)));
        if period > 0
            responseFeatures.naturalFrequency(i) = 1/period;
        end
        
        % Estimate damping from decay envelope
        [peaks, ~] = findpeaks(abs(postSegment));
        if length(peaks) >= 2
            logDec = log(peaks(1)/peaks(2));
            responseFeatures.dampingRatio(i) = logDec / sqrt((2*pi)^2 + logDec^2);
        end
    end
    
    % Rise time (10% to 90% of final value)
    finalValue = mean(postSegment(max(1, end-round(0.1/dt)):end));
    val10 = 0.1 * (finalValue - postSegment(1)) + postSegment(1);
    val90 = 0.9 * (finalValue - postSegment(1)) + postSegment(1);
    
    if postSegment(1) < finalValue % rising
        idx10 = find(postSegment > val10, 1, 'first');
        idx90 = find(postSegment > val90, 1, 'first');
    else % falling
        idx10 = find(postSegment < val10, 1, 'first');
        idx90 = find(postSegment < val90, 1, 'first');
    end
    
    if ~isempty(idx10) && ~isempty(idx90) && idx90 > idx10
        responseFeatures.riseTime90(i) = timePost(idx90) - timePost(idx10);
    end
    
    % Settling time (2% and 5% criteria)
    threshold2pct = 0.02 * abs(postSegment(1));
    threshold5pct = 0.05 * abs(postSegment(1));
    
    settleIdx2 = find(abs(postSegment) < threshold2pct, 1, 'first');
    settleIdx5 = find(abs(postSegment) < threshold5pct, 1, 'first');
    
    if ~isempty(settleIdx2)
        responseFeatures.settlingTime2pct(i) = timePost(settleIdx2);
    else
        responseFeatures.settlingTime2pct(i) = params.postEventWindow;
    end
    if ~isempty(settleIdx5)
        responseFeatures.settlingTime5pct(i) = timePost(settleIdx5);
    else
        responseFeatures.settlingTime5pct(i) = params.postEventWindow;
    end
    
    % Overshoot (if crosses zero)
    if sign(postSegment(1)) * sign(finalValue) < 0
        [maxVal, maxIdx] = max(abs(postSegment));
        responseFeatures.overshoot(i) = maxVal;
        responseFeatures.peakTime(i) = timePost(maxIdx);
    else
        responseFeatures.overshoot(i) = 0;
    end
    
    % Peak recovery
    responseFeatures.peakRecovery(i) = max(abs(postSegment));
    
    % Error metrics
    responseFeatures.rmsError(i) = rms(postSegment);
    responseFeatures.mae(i) = mean(abs(postSegment));
    responseFeatures.iae(i) = trapz(timePost, abs(postSegment));
    responseFeatures.ise(i) = trapz(timePost, postSegment.^2);
    responseFeatures.itae(i) = trapz(timePost, timePost .* abs(postSegment));
    
    % Recovery rate (average rate of return to zero in first 5 seconds)
    recoveryWindow = min(round(5/dt), length(postSegment));
    if recoveryWindow > 1
        responseFeatures.recoveryRate(i) = abs(postSegment(1) - postSegment(recoveryWindow)) / (recoveryWindow * dt);
    end
    
    % Frequency domain analysis
    if length(postSegment) > 64
        [pxx, f] = pwelch(postSegment, [], [], [], params.samplingRate);
        
        % Dominant frequency
        [~, maxIdx] = max(pxx);
        responseFeatures.dominantFrequency(i) = f(maxIdx);
        
        % Spectral power in frequency bands
        for b = 1:size(params.frequencyBands, 1)
            bandIdx = f >= params.frequencyBands(b,1) & f <= params.frequencyBands(b,2);
            if b == 1
                responseFeatures.spectralPower_lowFreq(i) = sum(pxx(bandIdx));
            elseif b == 2
                responseFeatures.spectralPower_midFreq(i) = sum(pxx(bandIdx));
            elseif b == 3
                responseFeatures.spectralPower_highFreq(i) = sum(pxx(bandIdx));
            end
        end
    end
end

fprintf('  Extracted %d response features\n\n', length(responseFeatureNames));

%% ========================================================================
%  SECTION 4: ADVANCED EXTERNAL SIGNAL FEATURE EXTRACTION
%  ========================================================================

fprintf('Step 3: Extracting advanced external signal features...\n');

externalFields = fieldnames(data.external);
numExternal = length(externalFields);

externalFeatures = struct();

% --- FIX: Helper function 'initializeExternalFeature' is inlined ---
% --- This correctly pre-allocates the struct fields. ---
for j = 1:numExternal
    signalName = externalFields{j};
    
    externalFeatures.([signalName '_atPowerOn']) = nan(numEvents, 1);
    externalFeatures.([signalName '_derivative_atPowerOn']) = nan(numEvents, 1);
    externalFeatures.([signalName '_meanDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_stdDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_minDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_maxDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_rangeDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_trendDuringOff']) = nan(numEvents, 1);
    externalFeatures.([signalName '_volatility']) = nan(numEvents, 1);
    
    % Time-lagged features
    for lagIdx = 1:length(params.timeLags)
        lag = params.timeLags(lagIdx);
        lagName = sprintf('%s_lag%.1fs', signalName, lag);
        lagName = strrep(lagName, '.', 'p');
        externalFeatures.(lagName) = nan(numEvents, 1);
    end
end
% --- End of inlined helper ---

for j = 1:numExternal
    signalName = externalFields{j};
    signal = data.external.(signalName);
    
    % Compute signal derivative
    signalDerivative = [0; diff(signal)] * params.samplingRate;
    
    for i = 1:numEvents
        idx = events(i).index;
        offStartIdx = events(i).offStartIdx;
        
        % Value at power-on
        externalFeatures.([signalName '_atPowerOn'])(i) = signal(idx);
        externalFeatures.([signalName '_derivative_atPowerOn'])(i) = signalDerivative(idx);
        
        % Time-lagged values
        for lagIdx = 1:length(params.timeLags)
            lag = params.timeLags(lagIdx);
            lagSamples = round(lag / dt);
            lagIdx_actual = max(1, idx - lagSamples);
            
            lagName = sprintf('%s_lag%.1fs', signalName, lag);
            lagName = strrep(lagName, '.', 'p');
            externalFeatures.(lagName)(i) = signal(lagIdx_actual);
        end
        
        % Features during off period
        if offStartIdx < idx - 1
            offSegment = signal(offStartIdx:idx-1);
            
            externalFeatures.([signalName '_meanDuringOff'])(i) = mean(offSegment);
            externalFeatures.([signalName '_stdDuringOff'])(i) = std(offSegment);
            externalFeatures.([signalName '_minDuringOff'])(i) = min(offSegment);
            externalFeatures.([signalName '_maxDuringOff'])(i) = max(offSegment);
            externalFeatures.([signalName '_rangeDuringOff'])(i) = range(offSegment);
            
            % Volatility (std of differences)
            if length(offSegment) > 1
                externalFeatures.([signalName '_volatility'])(i) = std(diff(offSegment));
            end
            
            % Trend during off period
            if length(offSegment) > 2
                p = polyfit(1:length(offSegment), offSegment', 1);
                externalFeatures.([signalName '_trendDuringOff'])(i) = p(1);
            end
        end
    end
end

externalFeatureNames = fieldnames(externalFeatures);
fprintf('  Extracted %d external signal features\n\n', length(externalFeatureNames));

%% ========================================================================
%  SECTION 5: INTERACTION FEATURES
%  ========================================================================

fprintf('Step 4: Computing interaction features...\n');

% Calculate variance of each external signal at power-on
externalVarAtPowerOn = zeros(numExternal, 1);
for j = 1:numExternal
    signalName = externalFields{j};
    featName = [signalName '_atPowerOn'];
    if isfield(externalFeatures, featName)
        externalVarAtPowerOn(j) = var(externalFeatures.(featName), 'omitnan');
    end
end

[~, topVarIdx] = sort(externalVarAtPowerOn, 'descend');
numInteractions = min(3, numExternal); % Top 3 most variable signals

% Create pairwise interactions
interactionCount = 0;
for j1 = 1:numInteractions
    for j2 = j1+1:numInteractions
        sig1Name = externalFields{topVarIdx(j1)};
        sig2Name = externalFields{topVarIdx(j2)};
        
        feat1Name = [sig1Name '_atPowerOn'];
        feat2Name = [sig2Name '_atPowerOn'];
        
        if isfield(externalFeatures, feat1Name) && isfield(externalFeatures, feat2Name)
            interactionName = sprintf('interaction_%s_x_%s', sig1Name, sig2Name);
            externalFeatures.(interactionName) = externalFeatures.(feat1Name) .* externalFeatures.(feat2Name);
            interactionCount = interactionCount + 1;
        end
    end
end

% Get updated list of external feature names
externalFeatureNames = fieldnames(externalFeatures);
fprintf('  Created %d interaction features\n\n', interactionCount);

%% ========================================================================
%  SECTION 6: DATA QUALITY AND OUTLIER DETECTION
%  ========================================================================

fprintf('Step 5: Detecting and handling outliers...\n');

% Combine all features
allFeatureNames = [responseFeatureNames'; externalFeatureNames]; %% <-- FIX: Transpose responseFeatureNames
allFeatures = zeros(numEvents, length(allFeatureNames));

for f = 1:length(responseFeatureNames)
    allFeatures(:, f) = responseFeatures.(responseFeatureNames{f});
end

for f = 1:length(externalFeatureNames)
    allFeatures(:, f + length(responseFeatureNames)) = externalFeatures.(externalFeatureNames{f});
end

% Detect outliers using MAD or IQR method
outlierMask = false(numEvents, length(allFeatureNames));

for f = 1:size(allFeatures, 2)
    x = allFeatures(:, f);
    x = x(isfinite(x));
    
    if isempty(x)
        continue;
    end
    
    if strcmpi(params.outlierMethod, 'MAD')
        medianX = median(x);
        madVal = median(abs(x - medianX));
        if madVal > 0
            mad_scaled = madVal * 1.4826;
            outlierMask(:, f) = abs(allFeatures(:, f) - medianX) > params.outlierThreshold * mad_scaled;
        end
    elseif strcmpi(params.outlierMethod, 'IQR')
        q25 = prctile(x, 25);
        q75 = prctile(x, 75);
        iqrVal = q75 - q25;
        if iqrVal > 0
            outlierMask(:, f) = allFeatures(:, f) < (q25 - params.outlierThreshold * iqrVal) | ...
                                allFeatures(:, f) > (q75 + params.outlierThreshold * iqrVal);
        end
    end
end

% Identify events with excessive outliers (>20% of features are outliers)
outlierEventMask = sum(outlierMask, 2) > 0.2 * size(allFeatures, 2);
numOutlierEvents = sum(outlierEventMask);

fprintf('  Detected %d events with excessive outliers (%.1f%%)\n', ...
        numOutlierEvents, 100*numOutlierEvents/numEvents);
fprintf('  These events will be flagged but included in analysis\n\n');

%% ========================================================================
%  SECTION 7: METHOD 1 - CORRELATION ANALYSIS
%  ========================================================================

fprintf('Step 6: Performing correlation analysis...\n');

correlationResults = struct();

%% <-- IMPROVEMENT: Add new robust features to the analysis list
% Response metrics to analyze (select key dynamic metrics)
responseMetricsToAnalyze = {'valueAtPowerOn', 'absValueAtPowerOn', ...
                             'timeConstant', 'settlingTime2pct', ...
                             'recoveryRate', 'iae', 'peakRecovery', ...
                             'finalSettlingValue', 'finalSlope', 'timeConstant_R2'};

for m = 1:length(responseMetricsToAnalyze)
    metricName = responseMetricsToAnalyze{m};
    
    if ~isfield(responseFeatures, metricName)
        continue;
    end
    
    responseData = responseFeatures.(metricName);
    validResponseIdx = isfinite(responseData) & ~outlierEventMask;
    
    correlationResults.(metricName) = struct();
    
    for corrType = 1:length(params.correlationMethods)
        corrMethod = params.correlationMethods{corrType};
        
        corrResults = struct();
        corrResults.correlations = nan(length(externalFeatureNames), 1);
        corrResults.pValues = ones(length(externalFeatureNames), 1);
        corrResults.ciLower = nan(length(externalFeatureNames), 1);
        corrResults.ciUpper = nan(length(externalFeatureNames), 1);
        
        for f = 1:length(externalFeatureNames)
            externalData = externalFeatures.(externalFeatureNames{f});
            validIdx = validResponseIdx & isfinite(externalData);
            
            if sum(validIdx) > 10
                % Compute correlation
                [r, p] = corr(responseData(validIdx), externalData(validIdx), ...
                              'Type', corrMethod, 'Rows', 'complete');
                
                corrResults.correlations(f) = r;
                corrResults.pValues(f) = p;
                
                % Bootstrap confidence intervals
                if params.numBootstrap > 0
                    try
                        bootR = bootstrp(params.numBootstrap, ...
                                        @(x,y) corr(x, y, 'Type', corrMethod, 'Rows', 'complete'), ...
                                        responseData(validIdx), externalData(validIdx));
                        corrResults.ciLower(f) = prctile(bootR, 2.5);
                        corrResults.ciUpper(f) = prctile(bootR, 97.5);
                    catch
                        % bootstrap can fail with NaNs, though 'complete' should handle
                    end
                end
            end
        end
        
        % Multiple testing correction
        if strcmpi(params.multipleTestCorrection, 'Bonferroni')
            corrResults.pValues_adjusted = min(1, corrResults.pValues * length(externalFeatureNames));
        elseif strcmpi(params.multipleTestCorrection, 'FDR')
            [~, ~, ~, corrResults.pValues_adjusted] = fdr_bh(corrResults.pValues, params.alphaLevel);
        else
            corrResults.pValues_adjusted = corrResults.pValues;
        end
        
        % Sort by absolute correlation
        [~, sortIdx] = sort(abs(corrResults.correlations), 'descend');
        corrResults.sortedIndices = sortIdx;
        
        correlationResults.(metricName).(corrMethod) = corrResults;
    end
end

fprintf('  Correlation analysis complete\n\n');

%% ========================================================================
%  SECTION 8: METHOD 2 - RANDOM FOREST FEATURE IMPORTANCE
%  ========================================================================

fprintf('Step 7: Performing Random Forest analysis...\n');

rfResults = struct();

% Prepare feature matrix
X_external = zeros(numEvents, length(externalFeatureNames));
for f = 1:length(externalFeatureNames)
    X_external(:, f) = externalFeatures.(externalFeatureNames{f});
end

% Remove rows/columns with too many NaNs
validRowIdx = sum(isnan(X_external), 2) < 0.3 * size(X_external, 2) & ~outlierEventMask;
validColIdx = sum(isnan(X_external), 1) < 0.3 * size(X_external, 1);

X_clean = X_external(validRowIdx, validColIdx);
featureNamesClean = externalFeatureNames(validColIdx);

% Impute remaining NaNs with median
for f = 1:size(X_clean, 2)
    nanIdx = isnan(X_clean(:, f));
    if any(nanIdx)
        medVal = median(X_clean(~nanIdx, f));
        if isfinite(medVal)
            X_clean(nanIdx, f) = medVal;
        else % if median is NaN (all data was NaN)
            X_clean(nanIdx, f) = 0; % impute with 0
        end
    end
end

% The new metrics from responseMetricsToAnalyze will be included here
for m = 1:length(responseMetricsToAnalyze)
    metricName = responseMetricsToAnalyze{m};
    
    if ~isfield(responseFeatures, metricName)
        continue;
    end
    
    y = responseFeatures.(metricName);
    y_clean = y(validRowIdx);
    validY = isfinite(y_clean);
    
    if sum(validY) < 20
        continue;
    end
    
    X_rf = X_clean(validY, :);
    y_rf = y_clean(validY);
    
    try
        % Train Random Forest
        rf = TreeBagger(params.rf.numTrees, X_rf, y_rf, ...
                       'Method', 'regression', ...
                       'OOBPredictorImportance', 'on', ...
                       'MinLeafSize', params.rf.minLeafSize, ...
                       'PredictorNames', featureNamesClean);
        
        % Feature importance
        importance = rf.OOBPermutedPredictorDeltaError;
        
        % Cross-validation R²
        yPred_oob = oobPredict(rf);
        r2_oob = 1 - sum((y_rf - yPred_oob).^2) / sum((y_rf - mean(y_rf)).^2);
        
        % Store results
        rfResults.(metricName) = struct();
        rfResults.(metricName).importance = importance;
        rfResults.(metricName).featureNames = featureNamesClean;
        rfResults.(metricName).r2_oob = r2_oob;
        [~, sortIdx] = sort(importance, 'descend');
        rfResults.(metricName).sortedIndices = sortIdx;
        
    catch ME
        warning('Random Forest failed for %s: %s', metricName, ME.message);
    end
end

fprintf('  Random Forest analysis complete\n\n');

%% ========================================================================
%  SECTION 9: METHOD 3 - MUTUAL INFORMATION
%  ========================================================================

fprintf('Step 8: Computing Mutual Information...\n');

miResults = struct();

% The new metrics from responseMetricsToAnalyze will be included here
for m = 1:length(responseMetricsToAnalyze)
    metricName = responseMetricsToAnalyze{m};
    
    if ~isfield(responseFeatures, metricName)
        continue;
    end
    
    responseData = responseFeatures.(metricName);
    validResponseIdx = isfinite(responseData) & ~outlierEventMask;
    
    miScores = nan(length(externalFeatureNames), 1);
    
    for f = 1:length(externalFeatureNames)
        externalData = externalFeatures.(externalFeatureNames{f});
        validIdx = validResponseIdx & isfinite(externalData);
        
        if sum(validIdx) > 10
            % Discretize for MI calculation
            y_disc = discretize(responseData(validIdx), 10);
            x_disc = discretize(externalData(validIdx), 10);
            
            % Remove any NaN from discretization
            validDisc = ~isnan(y_disc) & ~isnan(x_disc);
            
            if sum(validDisc) > 10
                % Compute mutual information
                miScores(f) = mi_discrete(y_disc(validDisc), x_disc(validDisc));
            end
        end
    end
    
    % Normalize MI scores
    maxMI = max(miScores);
    if maxMI > 0
        miScores = miScores / maxMI;
    end
    
    [~, sortIdx] = sort(miScores, 'descend');
    
    miResults.(metricName) = struct();
    miResults.(metricName).scores = miScores;
    miResults.(metricName).featureNames = externalFeatureNames;
    miResults.(metricName).sortedIndices = sortIdx;
end

fprintf('  Mutual Information analysis complete\n\n');

%% ========================================================================
%  SECTION 10: METHOD 4 - PARTIAL CORRELATION
%  ========================================================================

fprintf('Step 9: Computing Partial Correlations...\n');

partialCorrResults = struct();

% The new metrics from responseMetricsToAnalyze will be included here
for m = 1:length(responseMetricsToAnalyze)
    metricName = responseMetricsToAnalyze{m};
    
    if ~isfield(responseFeatures, metricName)
        continue;
    end
    
    responseData = responseFeatures.(metricName);
    
    % Prepare data matrix
    validIdx = validRowIdx & isfinite(responseData);
    y = responseData(validIdx);
    X = X_clean(validIdx(validRowIdx), :);
    
    if sum(validIdx) < 20 || size(X, 2) < 2
        continue;
    end
    
    % Compute partial correlations (correlation after controlling for all other variables)
    partialCorrs = nan(size(X, 2), 1);
    pValues = ones(size(X, 2), 1);
    
    for f = 1:size(X, 2)
        % Regress y on all other X variables
        X_others_idx = setdiff(1:size(X,2), f);
        X_others = X(:, X_others_idx);
        
        % Remove any NaN columns or constant columns
        validCols = var(X_others, 'omitnan') > 0;
        X_others = X_others(:, validCols);
        
        if size(X_others, 2) > 0 && rank(X_others) == size(X_others, 2)
            try
                % Residuals of y after accounting for other variables
                beta_y = X_others \ y;
                resid_y = y - X_others * beta_y;
                
                % Residuals of X(:,f) after accounting for other variables
                beta_x = X_others \ X(:, f);
                resid_x = X(:, f) - X_others * beta_x;
                
                % Correlation of residuals
                [r, p] = corr(resid_y, resid_x, 'Rows', 'complete');
                partialCorrs(f) = r;
                pValues(f) = p;
            catch
                % regression failed
            end
        end
    end
    
    [~, sortIdx] = sort(abs(partialCorrs), 'descend');
    
    partialCorrResults.(metricName) = struct();
    partialCorrResults.(metricName).correlations = partialCorrs;
    partialCorrResults.(metricName).pValues = pValues;
    partialCorrResults.(metricName).featureNames = featureNamesClean;
    partialCorrResults.(metricName).sortedIndices = sortIdx;
end

fprintf('  Partial correlation analysis complete\n\n');

%% ========================================================================
%  SECTION 11: METHOD 5 - GRANGER CAUSALITY (Temporal Precedence)
%  ========================================================================

fprintf('Step 10: Testing Granger Causality...\n');

grangerResults = struct();

% For each external signal, test if it Granger-causes the response
% This requires time-series analysis on the raw signals

maxLag = round(1 / dt); % Test up to 1 second of lag

for j = 1:numExternal
    signalName = externalFields{j};
    externalSignal = data.external.(signalName);
    
    % Test on full time series (not just events)
    validIdx = data.powerOn & isfinite(data.response) & isfinite(externalSignal);
    
    if sum(validIdx) < 100
        continue;
    end
    
    y = data.response(validIdx);
    x = externalSignal(validIdx);
    
    try
        % Perform Granger causality test
        [fStat, pValue] = granger_test(y, x, maxLag);
        
        grangerResults.(signalName) = struct();
        grangerResults.(signalName).fStatistic = fStat;
        grangerResults.(signalName).pValue = pValue;
        grangerResults.(signalName).significant = pValue < params.alphaLevel;
        
    catch ME
        warning('Granger test failed for %s: %s', signalName, ME.message);
    end
end

if ~isempty(fieldnames(grangerResults))
    fprintf('  Granger causality tests complete\n\n');
else
    fprintf('  Granger causality tests skipped (insufficient data)\n\n');
end

%% ========================================================================
%  SECTION 12: CONSOLIDATED RESULTS AND RANKING
%  ========================================================================

fprintf('Step 11: Consolidating results across methods...\n');

% Create consensus ranking by averaging normalized ranks across methods
consensusRanking = struct();

% The new metrics from responseMetricsToAnalyze will be included here
for m = 1:length(responseMetricsToAnalyze)
    metricName = responseMetricsToAnalyze{m};
    
    if ~isfield(responseFeatures, metricName)
        continue; % Skip if no data was generated
    end
    
    % Initialize ranks matrix
    numFeatures = length(externalFeatureNames);
    ranksMatrix = nan(numFeatures, 4); % 4 methods: Pearson, RF, MI, Partial
    
    % Pearson correlation ranks
    if isfield(correlationResults, metricName) && isfield(correlationResults.(metricName), 'Pearson')
        absCorr = abs(correlationResults.(metricName).Pearson.correlations);
        [~, ~, ranks] = unique(absCorr);
        ranksMatrix(:, 1) = max(ranks) - ranks + 1; % Reverse so 1 is best
    end
    
    % Random Forest importance ranks
    if isfield(rfResults, metricName)
        importance = nan(numFeatures, 1);
        featureNamesRF = rfResults.(metricName).featureNames;
        for f = 1:length(featureNamesRF)
            idx = find(strcmp(externalFeatureNames, featureNamesRF{f}), 1);
            if ~isempty(idx)
                importance(idx) = rfResults.(metricName).importance(f);
            end
        end
        [~, ~, ranks] = unique(importance);
        ranksMatrix(:, 2) = max(ranks) - ranks + 1;
    end
    
    % Mutual Information ranks
    if isfield(miResults, metricName)
        miScores = miResults.(metricName).scores;
        [~, ~, ranks] = unique(miScores);
        ranksMatrix(:, 3) = max(ranks) - ranks + 1;
    end
    
    % Partial correlation ranks
    if isfield(partialCorrResults, metricName)
        partialCorr = nan(numFeatures, 1);
        featureNamesPC = partialCorrResults.(metricName).featureNames;
        for f = 1:length(featureNamesPC)
            idx = find(strcmp(externalFeatureNames, featureNamesPC{f}), 1);
            if ~isempty(idx)
                partialCorr(idx) = abs(partialCorrResults.(metricName).correlations(f));
            end
        end
        [~, ~, ranks] = unique(partialCorr);
        ranksMatrix(:, 4) = max(ranks) - ranks + 1;
    end
    
    % Average ranks (ignoring NaNs from failed methods)
    avgRank = mean(ranksMatrix, 2, 'omitnan');
    [~, sortIdx] = sort(avgRank, 'asc'); % 'asc' because 1 is best rank
    
    consensusRanking.(metricName) = struct();
    consensusRanking.(metricName).avgRank = avgRank;
    consensusRanking.(metricName).sortedIndices = sortIdx;
    consensusRanking.(metricName).featureNames = externalFeatureNames;
    
end

fprintf('  Consensus ranking complete\n\n');

%% ========================================================================
%  SECTION 13: DISPLAY COMPREHENSIVE RESULTS
%  ========================================================================

fprintf('\n========================================\n');
fprintf('COMPREHENSIVE ANALYSIS RESULTS\n');
fprintf('========================================\n\n');

% Display for key metric: valueAtPowerOn
if isfield(consensusRanking, 'valueAtPowerOn')
    fprintf('=== PRIMARY METRIC: Response Value at Power-On ===\n\n');
    
    sortIdx = consensusRanking.valueAtPowerOn.sortedIndices(1:min(15, length(sortIdx)));
    
    fprintf('%-60s %8s %8s %8s %8s %10s\n', ...
            'External Feature', 'Pearson', 'RF Imp', 'MI', 'Partial', 'Avg Rank');
    fprintf('%s\n', repmat('-', 1, 104));
    
    for i = 1:length(sortIdx)
        idx = sortIdx(i);
        featName = externalFeatureNames{idx};
        
        % Get values from each method
        pearsonCorr = NaN;
        if isfield(correlationResults, 'valueAtPowerOn') && isfield(correlationResults.valueAtPowerOn, 'Pearson')
            pearsonCorr = correlationResults.valueAtPowerOn.Pearson.correlations(idx);
        end
        
        RFimp = NaN;
        if isfield(rfResults, 'valueAtPowerOn')
            rfFeats = rfResults.valueAtPowerOn.featureNames;
            rfIdx = find(strcmp(rfFeats, featName), 1);
            if ~isempty(rfIdx)
                impRaw = rfResults.valueAtPowerOn.importance(rfIdx);
                impMax = max(rfResults.valueAtPowerOn.importance);
                if impMax > 0
                    RFimp = impRaw / impMax; % Normalize for display
                end
            end
        end
        
        miScore = NaN;
        if isfield(miResults, 'valueAtPowerOn')
            miScore = miResults.valueAtPowerOn.scores(idx);
        end
        
        partialCorr = NaN;
        if isfield(partialCorrResults, 'valueAtPowerOn')
            pcFeats = partialCorrResults.valueAtPowerOn.featureNames;
            pcIdx = find(strcmp(pcFeats, featName), 1);
            if ~isempty(pcIdx)
                partialCorr = partialCorrResults.valueAtPowerOn.correlations(pcIdx);
            end
        end
        
        avgRank = consensusRanking.valueAtPowerOn.avgRank(idx);
        
        fprintf('%-60s %+7.3f %8.3f %7.3f %+7.3f %10.1f\n', ...
                truncateString(featName, 60), pearsonCorr, RFimp, miScore, partialCorr, avgRank);
    end
    fprintf('\n');
end

% Display Granger causality results
if ~isempty(fieldnames(grangerResults))
    fprintf('=== GRANGER CAUSALITY (Temporal Precedence) ===\n\n');
    fprintf('%-40s %10s %10s %s\n', 'External Signal', 'F-Stat', 'p-value', 'Significant');
    fprintf('%s\n', repmat('-', 1, 70));
    
    grangerFields = fieldnames(grangerResults);
    for j = 1:length(grangerFields)
        signalName = grangerFields{j};
        fStat = grangerResults.(signalName).fStatistic;
        pVal = grangerResults.(signalName).pValue;
        sig = grangerResults.(signalName).significant;
        
        sigStr = '';
        if sig
            sigStr = '***';
        end
        
        fprintf('%-40s %10.3f %10.4f %s\n', signalName, fStat, pVal, sigStr);
    end
    fprintf('\n');
end

% Display key statistics
fprintf('=== DATASET STATISTICS ===\n');
fprintf('Total events analyzed: %d\n', numEvents);
if numEvents > 0
    fprintf('Clean start events: %d\n', sum(~[events.isCascade]));
    fprintf('Cascade events: %d\n', sum([events.isCascade]));
    fprintf('Events flagged as outliers: %d (%.1f%%)\n', numOutlierEvents, 100*numOutlierEvents/numEvents);
    fprintf('Average off duration: %.2f ± %.2f seconds\n', mean([events.offDuration]), std([events.offDuration]));
end
fprintf('\n');

%% ========================================================================
%  SECTION 14: PUBLICATION-QUALITY VISUALIZATIONS
%  ========================================================================

fprintf('Step 12: Generating publication-quality visualizations...\n');

% Set default figure properties for publication quality
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextInterpreter', 'none');

%% Figure 1: Method Comparison Heatmap
fig1 = figure('Name', 'Method Comparison', 'Position', [100 100 1400 800], 'Visible', 'off');

subplot(2,2,1)
if isfield(correlationResults, 'valueAtPowerOn') && isfield(correlationResults.valueAtPowerOn, 'Pearson')
    plotMethodHeatmap(correlationResults.valueAtPowerOn.Pearson.correlations, ...
                     externalFeatureNames, 'Pearson Correlation');
end

subplot(2,2,2)
if isfield(rfResults, 'valueAtPowerOn')
    importance = nan(length(externalFeatureNames), 1);
    featureNamesRF = rfResults.valueAtPowerOn.featureNames;
    for f = 1:length(featureNamesRF)
        idx = find(strcmp(externalFeatureNames, featureNamesRF{f}), 1);
        if ~isempty(idx)
            importance(idx) = rfResults.valueAtPowerOn.importance(f);
        end
    end
    plotMethodHeatmap(importance / max(importance), externalFeatureNames, ...
                     'Random Forest Importance (Normalized)');
end

subplot(2,2,3)
if isfield(miResults, 'valueAtPowerOn')
    plotMethodHeatmap(miResults.valueAtPowerOn.scores, externalFeatureNames, ...
                     'Mutual Information (Normalized)');
end

subplot(2,2,4)
if isfield(partialCorrResults, 'valueAtPowerOn')
    partialCorr = nan(length(externalFeatureNames), 1);
    featureNamesPC = partialCorrResults.valueAtPowerOn.featureNames;
    for f = 1:length(featureNamesPC)
        idx = find(strcmp(externalFeatureNames, featureNamesPC{f}), 1);
        if ~isempty(idx)
            partialCorr(idx) = partialCorrResults.valueAtPowerOn.correlations(f);
        end
    end
    plotMethodHeatmap(partialCorr, externalFeatureNames, 'Partial Correlation');
end

sgtitle('Method Comparison: Value at Power-On', 'FontSize', 14, 'FontWeight', 'bold');

%% Figure 2: Consensus Top Features
fig2 = figure('Name', 'Consensus Top Features', 'Position', [150 150 1200 700], 'Visible', 'off');

if isfield(consensusRanking, 'valueAtPowerOn')
    sortIdx = consensusRanking.valueAtPowerOn.sortedIndices(1:min(20, length(sortIdx)));
    topRanks = consensusRanking.valueAtPowerOn.avgRank(sortIdx);
    topNames = consensusRanking.valueAtPowerOn.featureNames(sortIdx);
    
    barh(topRanks);
    set(gca, 'YTick', 1:length(topRanks), 'YTickLabel', cleanFeatureNames(topNames));
    xlabel('Average Rank Across Methods (Lower is Better)', 'FontSize', 12);
    ylabel('External Feature', 'FontSize', 12);
    title('Top 20 Features by Consensus Ranking', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'YDir', 'reverse');
end

%% Figure 3: Scatter Plots of Top Correlations
fig3 = figure('Name', 'Top Correlations Scatter', 'Position', [200 200 1600 900], 'Visible', 'off');

if isfield(consensusRanking, 'valueAtPowerOn')
    sortIdx = consensusRanking.valueAtPowerOn.sortedIndices(1:min(9, length(sortIdx)));
    
    for i = 1:length(sortIdx)
        subplot(3, 3, i);
        idx = sortIdx(i);
        
        responseData = responseFeatures.valueAtPowerOn;
        externalData = externalFeatures.(externalFeatureNames{idx});
        
        validIdx = isfinite(responseData) & isfinite(externalData) & ~outlierEventMask;
        
        if sum(validIdx) > 10
            scatter(externalData(validIdx), responseData(validIdx), 50, 'filled', ...
                   'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');

            % Add regression line
            p = polyfit(externalData(validIdx), responseData(validIdx), 1);
            xFit = linspace(min(externalData(validIdx)), max(externalData(validIdx)), 100);
            yFit = polyval(p, xFit);
            hold on;
            plot(xFit, yFit, 'r-', 'LineWidth', 2);

            xlabel(cleanFeatureName(externalFeatureNames{idx}), 'FontSize', 10);
            ylabel('Response at Power-On', 'FontSize', 10);

            % Add correlation coefficient
            if isfield(correlationResults, 'valueAtPowerOn') && isfield(correlationResults.valueAtPowerOn, 'Pearson')
                r = correlationResults.valueAtPowerOn.Pearson.correlations(idx);
                p_val = correlationResults.valueAtPowerOn.Pearson.pValues(idx);
                title(sprintf('r=%.3f, p=%.4f', r, p_val), 'FontSize', 10);
            end

            grid on;
        else
            title(sprintf('%s (Insufficient Data)', cleanFeatureName(externalFeatureNames{idx})), 'FontSize', 10);
        end
    end
end

sgtitle('Top Feature Correlations with Response at Power-On', 'FontSize', 14, 'FontWeight', 'bold');

%% Figure 4: Response Dynamics by Event Type
fig4 = figure('Name', 'Event Type Comparison', 'Position', [250 250 1400 600], 'Visible', 'off');

subplot(1,2,1)
cleanEvents = responseFeatures.valueAtPowerOn(~[events.isCascade] & ~outlierEventMask');
cascadeEvents = responseFeatures.valueAtPowerOn([events.isCascade] & ~outlierEventMask');

if ~isempty(cleanEvents) && ~isempty(cascadeEvents)
    boxplot([cleanEvents; cascadeEvents], ...
            [zeros(length(cleanEvents), 1); ones(length(cascadeEvents), 1)], ...
            'Labels', {'Clean Start', 'Cascade'});
    ylabel('Response Value at Power-On', 'FontSize', 12);
    title('Response Comparison by Event Type', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
else
    title('Response Comparison (Insufficient Data)', 'FontSize', 12);
end

subplot(1,2,2)
cleanSettling = responseFeatures.settlingTime2pct(~[events.isCascade] & ~outlierEventMask' & isfinite(responseFeatures.settlingTime2pct));
cascadeSettling = responseFeatures.settlingTime2pct([events.isCascade] & ~outlierEventMask' & isfinite(responseFeatures.settlingTime2pct));

if ~isempty(cleanSettling) && ~isempty(cascadeSettling)
    boxplot([cleanSettling; cascadeSettling], ...
            [zeros(length(cleanSettling), 1); ones(length(cascadeSettling), 1)], ...
            'Labels', {'Clean Start', 'Cascade'});
    ylabel('Settling Time (2%) [s]', 'FontSize', 12);
    title('Settling Time by Event Type', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
else
    title('Settling Time (Insufficient Data)', 'FontSize', 12);
end

%% Figure 5: Time Series Overview with Events
fig5 = figure('Name', 'Time Series Overview', 'Position', [300 300 1600 800], 'Visible', 'off');

subplot(3,1,1)
plot(data.time, data.response, 'b-', 'LineWidth', 1);
hold on;
if numEvents > 0
    eventTimes = [events.time];
    plot(eventTimes, zeros(size(eventTimes)), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    legend({'Response', 'Power-On Events'}, 'Location', 'best');
end
ylabel('Response Signal', 'FontSize', 11);
title('System Response with Power-On Events', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(3,1,2)
plot(data.time, data.powerOn, 'k-', 'LineWidth', 1.5);
ylabel('Power Status', 'FontSize', 11);
ylim([-0.1 1.1]);
title('System Power Status', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(3,1,3)
if numExternal > 0
    signalName = externalFields{1};
    plot(data.time, data.external.(signalName), 'Color', [0.2 0.6 0.2], 'LineWidth', 1);
    ylabel(cleanFeatureName(signalName), 'FontSize', 11);
    xlabel('Time [s]', 'FontSize', 11);
    title(['Example External Signal: ' cleanFeatureName(signalName)], 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
end

%% Figure 6: Feature Importance Comparison Across Metrics
fig6 = figure('Name', 'Feature Importance Across Metrics', 'Position', [350 350 1400 800], 'Visible', 'off');

if isfield(rfResults, 'valueAtPowerOn') && isfield(rfResults, 'settlingTime2pct')
    % Get top features from each metric
    topN = 15;
    
    featsValue = rfResults.valueAtPowerOn.featureNames;
    impValue = rfResults.valueAtPowerOn.importance;
    [impValue, sortIdxValue] = sort(impValue, 'descend');
    featsValue = featsValue(sortIdxValue(1:min(topN, length(sortIdxValue))));
    impValue = impValue(1:min(topN, length(impValue)));
    
    if isfield(rfResults, 'settlingTime2pct')
        featsSettling = rfResults.settlingTime2pct.featureNames;
        impSettling = rfResults.settlingTime2pct.importance;
        [impSettling, sortIdxSettling] = sort(impSettling, 'descend');
        featsSettling = featsSettling(sortIdxSettling(1:min(topN, length(sortIdxSettling))));
        impSettling = impSettling(1:min(topN, length(impSettling)));
        
        % Combine and plot
        subplot(1,2,1)
        barh(impValue);
        set(gca, 'YTick', 1:length(impValue), 'YTickLabel', cleanFeatureNames(featsValue));
        xlabel('RF Importance (OOB Permuted Error)', 'FontSize', 11);
        title('Value at Power-On', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        set(gca, 'YDir', 'reverse');
        
        subplot(1,2,2)
        barh(impSettling);
        set(gca, 'YTick', 1:length(impSettling), 'YTickLabel', cleanFeatureNames(featsSettling));
        xlabel('RF Importance (OOB Permuted Error)', 'FontSize', 11);
        title('Settling Time (2%)', 'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        set(gca, 'YDir', 'reverse');
    end
end

sgtitle('Random Forest Feature Importance Comparison', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('  Visualizations complete\n\n');

%% ========================================================================
%  SECTION 15: EXPORT RESULTS
%  ========================================================================

fprintf('Step 13: Exporting results...\n');

% Consolidate all results
results = struct();
results.metadata = struct();
results.metadata.numEvents = numEvents;
if numEvents > 0
    results.metadata.numCleanEvents = sum(~[events.isCascade]);
    results.metadata.numCascadeEvents = sum([events.isCascade]);
    results.metadata.numOutlierEvents = numOutlierEvents;
end
results.metadata.analysisDate = datestr(now);
results.metadata.params = params;

results.events = events;
results.responseFeatures = responseFeatures;
results.externalFeatures = externalFeatures;
results.correlationResults = correlationResults;
results.rfResults = rfResults;
results.miResults = miResults;
results.partialCorrResults = partialCorrResults;
results.grangerResults = grangerResults;
results.consensusRanking = consensusRanking;
results.outlierEventMask = outlierEventMask;

% Save to MAT file
save('publication_grade_analysis_results_v2.mat', 'results', '-v7.3');
fprintf('  Results saved to: publication_grade_analysis_results_v2.mat\n');

% Export summary table to CSV
if isfield(consensusRanking, 'valueAtPowerOn')
    exportSummaryTable(consensusRanking.valueAtPowerOn, correlationResults.valueAtPowerOn, ...
                      rfResults, miResults, partialCorrResults, externalFeatureNames);
    fprintf('  Summary table exported to: feature_importance_summary_v2.csv\n');
end

% Save figures
saveas(fig1, 'fig1_method_comparison_v2.png');
saveas(fig2, 'fig2_consensus_ranking_v2.png');
saveas(fig3, 'fig3_scatter_plots_v2.png');
saveas(fig4, 'fig4_event_types_v2.png');
saveas(fig5, 'fig5_time_series_overview_v2.png');
saveas(fig6, 'fig6_importance_comparison_v2.png');
fprintf('  Figures saved as PNG files\n\n');

fprintf('========================================\n');
fprintf('ANALYSIS COMPLETE!\n');
fprintf('========================================\n\n');

fprintf('Key Findings Summary:\n');
fprintf('--------------------\n');
if isfield(consensusRanking, 'valueAtPowerOn')
    sortIdx = consensusRanking.valueAtPowerOn.sortedIndices(1:min(5, length(sortIdx)));
    fprintf('Top 5 most influential external signals (lower rank is better):\n');
    for i = 1:length(sortIdx)
        idx = sortIdx(i);
        fprintf('  %d. %s (avg rank: %.1f)\n', i, ...
                externalFeatureNames{idx}, consensusRanking.valueAtPowerOn.avgRank(idx));
    end
end

% Close invisible figures
close(fig1);
close(fig2);
close(fig3);
close(fig4);
close(fig5);
close(fig6);

end % --- THIS ENDS THE MAIN FUNCTION ---

%% ========================================================================
%  HELPER FUNCTIONS (Now correctly scoped as local functions)
%  ========================================================================

    function plotMethodHeatmap(values, names, titleStr)
        % Plot top 30 features
        [sortedVals, sortIdx] = sort(abs(values), 'descend', 'MissingPlacement', 'last');
        topN = min(30, sum(isfinite(sortedVals)));
        
        if topN == 0
            title(titleStr, 'FontSize', 11, 'FontWeight', 'bold');
            text(0.5, 0.5, 'No Data', 'HorizontalAlignment', 'center', 'FontSize', 12);
            axis off;
            return;
        end
        
        imagesc(sortedVals(1:topN)');
        colormap(gca, parula);
        colorbar;
        set(gca, 'YTick', 1, 'YTickLabel', {''});
        set(gca, 'XTick', 1:topN, 'XTickLabel', cleanFeatureNames(names(sortIdx(1:topN))), ...
                'XTickLabelRotation', 90);
        title(titleStr, 'FontSize', 11, 'FontWeight', 'bold');
    end

    function cleanNames = cleanFeatureNames(names)
        cleanNames = cell(size(names));
        for i = 1:length(names)
            cleanNames{i} = cleanFeatureName(names{i});
        end
    end

    function cleanName = cleanFeatureName(name)
        cleanName = strrep(name, '_', ' ');
        if length(cleanName) > 50
            cleanName = [cleanName(1:47) '...'];
        end
    end

    function str = truncateString(str, maxLen)
        if length(str) > maxLen
            str = [str(1:maxLen-3) '...'];
        end
    end

    function exportSummaryTable(consensus, corrRes, rfRes, miRes, pcRes, featNames)
        sortIdx = consensus.sortedIndices(1:min(50, length(consensus.sortedIndices)));
        
        fid = fopen('feature_importance_summary_v2.csv', 'w');
        fprintf(fid, 'Rank,Feature,AvgRank,Pearson,Pearson_p_adj,RF_Importance_Norm,MI_Score,PartialCorr,PartialCorr_p\n');
        
        for i = 1:length(sortIdx)
            idx = sortIdx(i);
            featName = featNames{idx};
            avgRank = consensus.avgRank(idx);
            
            % Get values from each method
            pearsonCorr = NaN;
            pearsonP = NaN;
            if isfield(corrRes, 'Pearson')
                pearsonCorr = corrRes.Pearson.correlations(idx);
                pearsonP = corrRes.Pearson.pValues_adjusted(idx);
            end
            
            RFimp = NaN;
            if ~isempty(rfRes) && isfield(rfRes, 'valueAtPowerOn')
                rfFeats = rfRes.valueAtPowerOn.featureNames;
                rfIdx = find(strcmp(rfFeats, featName), 1);
                if ~isempty(rfIdx)
                    impRaw = rfRes.valueAtPowerOn.importance(rfIdx);
                    impMax = max(rfRes.valueAtPowerOn.importance);
                    if impMax > 0
                        RFimp = impRaw / impMax;
                    end
                end
            end
            
            miScore = NaN;
            if ~isempty(miRes) && isfield(miRes, 'valueAtPowerOn')
                miScore = miRes.valueAtPowerOn.scores(idx);
            end
            
            partialCorr = NaN;
            partialP = NaN;
            if ~isempty(pcRes) && isfield(pcRes, 'valueAtPowerOn')
                pcFeats = pcRes.valueAtPowerOn.featureNames;
                pcIdx = find(strcmp(pcFeats, featName), 1);
                if ~isempty(pcIdx)
                    partialCorr = pcRes.valueAtPowerOn.correlations(pcIdx);
                    partialP = pcRes.valueAtPowerOn.pValues(pcIdx);
                end
            end
            
            fprintf(fid, '%d,%s,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', ...
                    i, featName, avgRank, pearsonCorr, pearsonP, RFimp, miScore, ...
                    partialCorr, partialP);
        end
        
        fclose(fid);
    end

    function mi = mi_discrete(x, y)
        % Calculate mutual information for discrete variables
        % MI(X;Y) = H(X) + H(Y) - H(X,Y)
        
        n = length(x);
        
        % Joint probability
        % Use sparse matrix for efficiency
        pxy = sparse(x, y, 1, max(x), max(y));
        pxy = pxy / n;
        pxy_flat = pxy(pxy > 0);
        
        % Marginal probabilities
        px = sum(pxy, 2);
        py = sum(pxy, 1);
        
        px = px(px > 0);
        py = py(py > 0);
        
        % Entropies
        hx = -sum(px .* log2(px));
        hy = -sum(py .* log2(py));
        hxy = -sum(pxy_flat .* log2(pxy_flat));
        
        % Mutual information
        mi = hx + hy - hxy;
    end

    function [fStat, pValue] = granger_test(y, x, maxLag)
        % Simple Granger causality test
        % Tests if x Granger-causes y
        
        n_obs = length(y);
        n_reg = n_obs - maxLag; % Number of samples for regression
        
        % Unrestricted model: y regressed on lagged y and lagged x
        X_unrestricted = zeros(n_reg, 2 * maxLag);
        for lag = 1:maxLag
            X_unrestricted(:, lag) = y(maxLag - lag + 1:n_obs - lag);
            X_unrestricted(:, maxLag + lag) = x(maxLag - lag + 1:n_obs - lag);
        end
        y_target = y(maxLag + 1:end);
        
        % Restricted model: y regressed only on lagged y
        X_restricted = X_unrestricted(:, 1:maxLag);
        
        % Fit models
        beta_unrestricted = X_unrestricted \ y_target;
        beta_restricted = X_restricted \ y_target;
        
        % Calculate residual sum of squares
        rss_unrestricted = sum((y_target - X_unrestricted * beta_unrestricted).^2);
        rss_restricted = sum((y_target - X_restricted * beta_restricted).^2);
        
        % F-statistic
        p_restricted = maxLag;
        p_unrestricted = 2 * maxLag;
        
        fStat = ((rss_restricted - rss_unrestricted) / (p_unrestricted - p_restricted)) / ...
                (rss_unrestricted / (n_reg - p_unrestricted));
        
        % P-value from F-distribution
        df1 = p_unrestricted - p_restricted;
        df2 = n_reg - p_unrestricted;
        pValue = 1 - fcdf(fStat, df1, df2);
    end

    function [h, crit_p, adj_ci, adj_p] = fdr_bh(pvals, q)
        % fdr_bh: False Discovery Rate correction (Benjamini-Hochberg)
        %
        % Usage:
        % [h, crit_p, adj_ci, adj_p] = fdr_bh(pvals, q)
        %
        % Inputs:
        %   pvals       - Vector of p-values
        %   q           - False discovery rate level (e.g., 0.05)
        %
        % Outputs:
        %   h           - Vector of booleans (1=reject H0, 0=do not reject)
        %   crit_p      - The p-value threshold for significance
        %   adj_ci      - Adjusted critical p-value (not really used, legacy)
        %   adj_p       - Vector of FDR-adjusted p-values
        
        pvals = pvals(:);
        nanPvals = isnan(pvals);
        pvals(nanPvals) = 1; % Set NaNs to 1 so they don't affect sorting
        
        m = length(pvals);
        [sorted_p, sort_idx] = sort(pvals);
        
        % Calculate adjusted p-values
        adj_p = m * sorted_p ./ (1:m)';
        
        % Enforce monotonicity (make sure they are non-decreasing)
        for i = (m-1):-1:1
            adj_p(i) = min(adj_p(i), adj_p(i+1));
        end
        
        % "Unsort" adjusted p-values
        adj_p_unsorted = zeros(m,1);
        adj_p_unsorted(sort_idx) = adj_p;
        adj_p = adj_p_unsorted;
        
        % Find the largest p-value that is significant
        significant_p = sorted_p(sorted_p <= (1:m)'/m * q);
        if isempty(significant_p)
            crit_p = 0;
        else
            crit_p = max(significant_p);
        end
        
        h = adj_p <= q;
        adj_ci = crit_p; % for compatibility
        
        % Restore NaNs
        adj_p(nanPvals) = NaN;
        h(nanPvals) = NaN;
    end

    function flatStruct = flattenNestedStruct(nestedStruct, prefix)
        % Recursively flatten a nested struct into a flat struct with signal arrays
        %
        % Inputs:
        %   nestedStruct - Potentially nested struct containing signal data
        %   prefix       - String prefix for field names (default: '')
        %
        % Output:
        %   flatStruct   - Flat struct where all fields are numeric arrays
        %
        % Example:
        %   Input:  data.sensors.temperature = [1 2 3]
        %           data.sensors.pressure = [4 5 6]
        %           data.voltage = [7 8 9]
        %   Output: flat.sensors_temperature = [1 2 3]
        %           flat.sensors_pressure = [4 5 6]
        %           flat.voltage = [7 8 9]

        if nargin < 2
            prefix = '';
        end

        flatStruct = struct();

        if ~isstruct(nestedStruct)
            % If input is not a struct, return empty
            return;
        end

        fields = fieldnames(nestedStruct);

        for i = 1:length(fields)
            fieldName = fields{i};
            fieldValue = nestedStruct.(fieldName);

            % Create the flattened field name
            if isempty(prefix)
                flatFieldName = fieldName;
            else
                flatFieldName = [prefix '_' fieldName];
            end

            % Check if this field is a struct (nested) or a signal (array)
            if isstruct(fieldValue)
                % Recursively flatten nested struct
                nestedFlat = flattenNestedStruct(fieldValue, flatFieldName);
                nestedFields = fieldnames(nestedFlat);

                % Copy flattened fields to output
                for j = 1:length(nestedFields)
                    flatStruct.(nestedFields{j}) = nestedFlat.(nestedFields{j});
                end

            elseif isnumeric(fieldValue) || islogical(fieldValue)
                % This is a signal (numeric or logical array)
                % Add it to the flat structure
                flatStruct.(flatFieldName) = fieldValue;

            else
                % Skip non-numeric, non-struct fields (e.g., strings, cells)
                warning('Skipping field "%s" (type: %s) - only numeric/logical arrays supported', ...
                        flatFieldName, class(fieldValue));
            end
        end
    end

    function data = generateSyntheticData()
        % Generate realistic synthetic data for demonstration
        fprintf('Generating synthetic data...\n');
        
        dt = 0.01;
        t = (0:dt:2000)'; % 2000 seconds
        n = length(t);
        
        % Power-on signal with realistic outages
        powerOn = true(n, 1);
        numOutages = 30;
        for i = 1:numOutages
            outageStart = randi([500, n-5000]);
            outageDuration = randi([round(2/dt), round(10/dt)]); % 2-10 seconds
            powerOn(outageStart:min(outageStart+outageDuration, n)) = false;
        end
        
        % Response signal with realistic dynamics
        response = zeros(n, 1);
        controlGain = 0.5;
        tau = 3.0; % Time constant
        
        % External signals with different characteristics
        % DEMONSTRATION: Using NESTED STRUCTURE to show flexibility
        external = struct();

        % Nested under 'environmental' category
        external.environmental.temperature = 5 + 3 * smoothdata(randn(n,1), 'gaussian', 500);
        external.environmental.humidity = 50 + 10 * smoothdata(randn(n,1), 'gaussian', 800);

        % Nested under 'system' category
        external.system.voltage = 12 + 0.5 * smoothdata(randn(n,1), 'gaussian', 1000);
        external.system.current = 2 + 0.3 * smoothdata(randn(n,1), 'gaussian', 600);

        % Nested under 'sensors' category
        external.sensors.noise = smoothdata(randn(n,1), 'gaussian', 100);
        external.sensors.vibration = 0.5 * smoothdata(randn(n,1), 'gaussian', 300);

        % Top-level signal (not nested)
        external.reference_signal = 0.5 * smoothdata(randn(n,1), 'gaussian', 800);

        % Map to original variable names for simulation logic
        sig_A_drift = external.environmental.temperature;
        sig_B_dynamics = external.system.voltage / 12; % Normalize to ~1
        sig_C_distractor = external.sensors.noise;
        sig_F_settling = external.reference_signal;
        
        % Simulate response with realistic control dynamics
        for i = 2:n
            if powerOn(i)
                % Control loop active - exponential decay
                % Dynamic tau based on Signal B
                current_tau = tau / external.sig_B_dynamics(i);
                
                % Disturbance from signal C
                disturbance = 0.01 * external.sig_C_distractor(i);
                
                % Non-zero setpoint from Signal F
                setpoint = external.sig_F_settling(i);
                
                % First-order system dynamics (decaying to setpoint)
                response(i) = response(i-1) * (1 - dt/current_tau) + ...
                             (setpoint/current_tau + disturbance) * dt;
            else
                % Control loop inactive - drift based on external signal A
                drift = 0.1 * external.sig_A_drift(i);
                response(i) = response(i-1) + drift*dt + 0.01*randn;
            end
        end
        
        data.time = t;
        data.response = response + 0.01*randn(n,1); % Add measurement noise
        data.powerOn = powerOn;
        data.external = external;
        
        fprintf('Synthetic data generated (n=%d samples, %.1f sec duration)\n\n', n, t(end));
    end