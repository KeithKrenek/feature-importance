function [results, figHandle] = analyze_feature_importance(dataTable, varargin)
% ANALYZE_FEATURE_IMPORTANCE - Quantify feature importance using machine learning
%
% This function analyzes which features (predictor variables) are most strongly
% correlated with a target variable using multiple machine learning approaches.
%
% SYNTAX:
%   results = analyze_feature_importance(dataTable)
%   results = analyze_feature_importance(dataTable, 'FeatureColumns', featureCols, 'TargetColumn', targetCol)
%   [results, figHandle] = analyze_feature_importance(...)
%
% INPUTS:
%   dataTable     - MATLAB table containing the data
%
% OPTIONAL NAME-VALUE PAIRS:
%   'FeatureColumns'  - Cell array of column names or numeric indices for features
%                       If empty, user will be prompted to select
%   'TargetColumn'    - Column name (char/string) or numeric index for target
%                       If empty, user will be prompted to select
%   'Method'          - Analysis method: 'auto' (default), 'all', 'randomforest', 
%                       'relief', 'correlation', 'mutualinfo'
%   'NumTrees'        - Number of trees for Random Forest (default: 100)
%   'Normalize'       - Boolean, normalize features (default: true)
%   'PlotResults'     - Boolean, create visualization (default: true)
%   'Verbose'         - Boolean, display progress messages (default: true)
%
% OUTPUTS:
%   results    - Structure containing:
%                .FeatureNames     - Names of features analyzed
%                .TargetName       - Name of target variable
%                .ProblemType      - 'classification' or 'regression'
%                .Methods          - Cell array of methods used
%                .Scores           - Matrix of importance scores (features x methods)
%                .Ranks            - Matrix of feature ranks (features x methods)
%                .CombinedScore    - Average normalized score across methods
%                .CombinedRank     - Final ranking based on combined score
%                .ModelAccuracy    - Accuracy/R² from Random Forest model
%   figHandle  - Handle to the generated figure (if PlotResults is true)
%
% EXAMPLE 1: Interactive mode
%   load fisheriris
%   irisTable = array2table([meas, grp2idx(species)], ...
%       'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'});
%   results = analyze_feature_importance(irisTable);
%
% EXAMPLE 2: Programmatic mode
%   results = analyze_feature_importance(irisTable, ...
%       'FeatureColumns', 1:4, ...
%       'TargetColumn', 'Species', ...
%       'Method', 'all');
%
% EXAMPLE 3: Regression problem
%   load carsmall
%   carTable = table(Acceleration, Cylinders, Displacement, Horsepower, Weight, MPG);
%   results = analyze_feature_importance(carTable, ...
%       'FeatureColumns', 1:5, ...
%       'TargetColumn', 'MPG');
%
% NOTES:
%   - Automatically detects classification vs regression based on target variable
%   - Handles missing data by removing incomplete rows
%   - For classification, target should be categorical or have few unique values
%   - Uses ensemble methods for robust importance estimation
%
% Author: KK
% Version: 1.0
% Date: 2025

%% Parse Inputs
p = inputParser;
addRequired(p, 'dataTable', @(x) istable(x) && height(x) > 0);
addParameter(p, 'FeatureColumns', [], @(x) isempty(x) || iscell(x) || isnumeric(x));
addParameter(p, 'TargetColumn', [], @(x) isempty(x) || ischar(x) || isstring(x) || isnumeric(x));
addParameter(p, 'Method', 'auto', @(x) any(strcmpi(x, {'auto','all','randomforest','relief','correlation','mutualinfo'})));
addParameter(p, 'NumTrees', 100, @(x) isnumeric(x) && x > 0);
addParameter(p, 'Normalize', true, @islogical);
addParameter(p, 'PlotResults', true, @islogical);
addParameter(p, 'Verbose', true, @islogical);

parse(p, dataTable, varargin{:});

featureCols = p.Results.FeatureColumns;
targetCol = p.Results.TargetColumn;
method = lower(p.Results.Method);
numTrees = p.Results.NumTrees;
normalizeData = p.Results.Normalize;
plotResults = p.Results.PlotResults;
verbose = p.Results.Verbose;

%% Column Selection (Interactive if not specified)
if isempty(featureCols) || isempty(targetCol)
    if verbose
        fprintf('\n=== FEATURE IMPORTANCE ANALYZER ===\n');
        fprintf('Available columns in the table:\n');
        for i = 1:width(dataTable)
            fprintf('  [%d] %s\n', i, dataTable.Properties.VariableNames{i});
        end
    end
    
    % Select target column
    if isempty(targetCol)
        targetIdx = input('\nEnter the column number for the TARGET variable: ');
        if targetIdx < 1 || targetIdx > width(dataTable)
            error('Invalid target column index.');
        end
        targetCol = targetIdx;
    end
    
    % Select feature columns
    if isempty(featureCols)
        fprintf('\nEnter feature column numbers (e.g., [1 2 3] or 1:5).\n');
        fprintf('Press Enter to use all columns except target.\n');
        featureInput = input('Feature columns: ', 's');
        
        if isempty(featureInput)
            % Use all columns except target
            featureCols = setdiff(1:width(dataTable), targetCol);
        else
            featureCols = str2num(featureInput); %#ok<ST2NM>
            if isempty(featureCols)
                error('Invalid feature column specification.');
            end
        end
    end
end

%% Convert column specifications to indices and names
if isnumeric(targetCol)
    targetIdx = targetCol;
    targetName = dataTable.Properties.VariableNames{targetIdx};
else
    targetName = char(targetCol);
    targetIdx = find(strcmp(dataTable.Properties.VariableNames, targetName));
    if isempty(targetIdx)
        error('Target column "%s" not found in table.', targetName);
    end
end

if iscell(featureCols)
    featureNames = featureCols;
    featureIdx = zeros(1, length(featureNames));
    for i = 1:length(featureNames)
        idx = find(strcmp(dataTable.Properties.VariableNames, featureNames{i}));
        if isempty(idx)
            error('Feature column "%s" not found in table.', featureNames{i});
        end
        featureIdx(i) = idx;
    end
else
    featureIdx = featureCols;
    featureNames = dataTable.Properties.VariableNames(featureIdx);
end

% Validate no overlap
if any(featureIdx == targetIdx)
    error('Feature columns cannot include the target column.');
end

%% Extract and Prepare Data
if verbose
    fprintf('\n--- Data Preparation ---\n');
    fprintf('Features: %s\n', strjoin(featureNames, ', '));
    fprintf('Target: %s\n', targetName);
end

% Extract data
X_raw = dataTable{:, featureIdx};
y_raw = dataTable{:, targetIdx};

% Handle missing data
completeRows = all(~isnan([X_raw, y_raw]), 2);
if ~iscell(y_raw)
    completeRows = completeRows & ~isnan(y_raw);
end

if sum(~completeRows) > 0
    if verbose
        fprintf('Removing %d rows with missing data (%.1f%% of data).\n', ...
            sum(~completeRows), 100*sum(~completeRows)/length(completeRows));
    end
    X_raw = X_raw(completeRows, :);
    y_raw = y_raw(completeRows, :);
end

% Determine problem type
if iscategorical(y_raw) || ischar(y_raw) || iscell(y_raw) || isstring(y_raw)
    problemType = 'classification';
    if ~iscategorical(y_raw)
        y = categorical(y_raw);
    else
        y = y_raw;
    end
    numClasses = length(categories(y));
elseif length(unique(y_raw)) <= 20 && length(unique(y_raw)) < 0.1 * length(y_raw)
    % Likely classification if few unique values
    problemType = 'classification';
    y = categorical(y_raw);
    numClasses = length(categories(y));
else
    problemType = 'regression';
    y = y_raw;
    numClasses = 0;
end

if verbose
    fprintf('Problem type: %s\n', upper(problemType));
    if strcmp(problemType, 'classification')
        fprintf('Number of classes: %d\n', numClasses);
    end
    fprintf('Sample size: %d observations, %d features\n', size(X_raw, 1), size(X_raw, 2));
end

% Normalize features if requested
if normalizeData
    X = normalize(X_raw, 'range');
else
    X = X_raw;
end

%% Determine Which Methods to Use
if strcmp(method, 'auto')
    if strcmp(problemType, 'classification')
        methodsToUse = {'randomforest', 'relief', 'mutualinfo'};
    else
        methodsToUse = {'randomforest', 'relief', 'correlation'};
    end
elseif strcmp(method, 'all')
    if strcmp(problemType, 'classification')
        methodsToUse = {'randomforest', 'relief', 'mutualinfo'};
    else
        methodsToUse = {'randomforest', 'relief', 'correlation', 'mutualinfo'};
    end
else
    methodsToUse = {method};
end

% Check for incompatible method-problem combinations
if strcmp(problemType, 'classification') && ismember('correlation', methodsToUse)
    warning('Correlation method is designed for regression. Removing from analysis.');
    methodsToUse(strcmp(methodsToUse, 'correlation')) = [];
end

%% Initialize Results Storage
numFeatures = length(featureNames);
numMethods = length(methodsToUse);
scores = zeros(numFeatures, numMethods);
methodLabels = cell(1, numMethods);

%% Method 1: Random Forest Feature Importance
if ismember('randomforest', methodsToUse)
    methodIdx = find(strcmp(methodsToUse, 'randomforest'));
    methodLabels{methodIdx} = 'Random Forest';
    
    if verbose
        fprintf('\n--- Random Forest Analysis ---\n');
        fprintf('Training ensemble with %d trees...\n', numTrees);
    end
    
    try
        if strcmp(problemType, 'classification')
            rfModel = TreeBagger(numTrees, X, y, ...
                'OOBPredictorImportance', 'on', ...
                'Method', 'classification', ...
                'NumPredictorsToSample', 'all');
            
            % Calculate OOB error
            oobError = oobError(rfModel);
            modelAccuracy = 1 - oobError(end);
            
            if verbose
                fprintf('OOB Classification Accuracy: %.2f%%\n', modelAccuracy * 100);
            end
        else
            rfModel = TreeBagger(numTrees, X, y, ...
                'OOBPredictorImportance', 'on', ...
                'Method', 'regression', ...
                'NumPredictorsToSample', 'all');
            
            % Calculate R²
            yhat = oobPredict(rfModel);
            ss_res = sum((y - yhat).^2);
            ss_tot = sum((y - mean(y)).^2);
            modelAccuracy = 1 - ss_res/ss_tot;
            
            if verbose
                fprintf('OOB R² Score: %.4f\n', modelAccuracy);
            end
        end
        
        scores(:, methodIdx) = rfModel.OOBPermutedPredictorDeltaError;
        
        if verbose
            fprintf('Random Forest importance computed.\n');
        end
    catch ME
        warning('Random Forest failed: %s', ME.message);
        scores(:, methodIdx) = NaN;
        modelAccuracy = NaN;
    end
else
    modelAccuracy = NaN;
end

%% Method 2: ReliefF Algorithm
if ismember('relief', methodsToUse)
    methodIdx = find(strcmp(methodsToUse, 'relief'));
    methodLabels{methodIdx} = 'ReliefF';
    
    if verbose
        fprintf('\n--- ReliefF Analysis ---\n');
    end
    
    try
        if strcmp(problemType, 'classification')
            [reliefScores, ~] = relieff(X, y, 10);
        else
            [reliefScores, ~] = relieff(X, y, 10);
        end
        
        scores(:, methodIdx) = reliefScores;
        
        if verbose
            fprintf('ReliefF importance computed.\n');
        end
    catch ME
        warning('ReliefF failed: %s', ME.message);
        scores(:, methodIdx) = NaN;
    end
end

%% Method 3: Correlation Analysis (Regression Only)
if ismember('correlation', methodsToUse)
    methodIdx = find(strcmp(methodsToUse, 'correlation'));
    methodLabels{methodIdx} = 'Correlation';
    
    if verbose
        fprintf('\n--- Correlation Analysis ---\n');
    end
    
    try
        correlations = zeros(numFeatures, 1);
        for i = 1:numFeatures
            R = corrcoef(X(:, i), y);
            correlations(i) = abs(R(1, 2));
        end
        
        scores(:, methodIdx) = correlations;
        
        if verbose
            fprintf('Correlation coefficients computed.\n');
        end
    catch ME
        warning('Correlation analysis failed: %s', ME.message);
        scores(:, methodIdx) = NaN;
    end
end

%% Method 4: Mutual Information
if ismember('mutualinfo', methodsToUse)
    methodIdx = find(strcmp(methodsToUse, 'mutualinfo'));
    methodLabels{methodIdx} = 'Mutual Information';
    
    if verbose
        fprintf('\n--- Mutual Information Analysis ---\n');
    end
    
    try
        miScores = zeros(numFeatures, 1);
        
        for i = 1:numFeatures
            if strcmp(problemType, 'classification')
                % For classification, discretize the feature
                x_discrete = discretize(X(:, i), 10);
                miScores(i) = calculate_mutual_information(x_discrete, double(y));
            else
                % For regression, discretize both variables
                x_discrete = discretize(X(:, i), 10);
                y_discrete = discretize(y, 10);
                miScores(i) = calculate_mutual_information(x_discrete, y_discrete);
            end
        end
        
        scores(:, methodIdx) = miScores;
        
        if verbose
            fprintf('Mutual information computed.\n');
        end
    catch ME
        warning('Mutual Information failed: %s', ME.message);
        scores(:, methodIdx) = NaN;
    end
end

%% Combine Results
if verbose
    fprintf('\n--- Computing Combined Rankings ---\n');
end

% Normalize scores to [0, 1] range for each method
scoresNorm = zeros(size(scores));
for i = 1:numMethods
    validScores = scores(~isnan(scores(:, i)), i);
    if ~isempty(validScores)
        minScore = min(validScores);
        maxScore = max(validScores);
        if maxScore > minScore
            scoresNorm(:, i) = (scores(:, i) - minScore) / (maxScore - minScore);
        else
            scoresNorm(:, i) = ones(numFeatures, 1);
        end
    else
        scoresNorm(:, i) = NaN;
    end
end

% Compute combined score (average of normalized scores)
combinedScore = mean(scoresNorm, 2, 'omitnan');

% Compute ranks
ranks = zeros(size(scores));
for i = 1:numMethods
    [~, sortIdx] = sort(scores(:, i), 'descend');
    ranks(sortIdx, i) = 1:numFeatures;
end

% Combined rank based on combined score
[~, sortIdx] = sort(combinedScore, 'descend');
combinedRank = zeros(numFeatures, 1);
combinedRank(sortIdx) = 1:numFeatures;

%% Create Results Structure
results = struct();
results.FeatureNames = featureNames;
results.TargetName = targetName;
results.ProblemType = problemType;
results.Methods = methodLabels;
results.Scores = scores;
results.Ranks = ranks;
results.NormalizedScores = scoresNorm;
results.CombinedScore = combinedScore;
results.CombinedRank = combinedRank;
results.ModelAccuracy = modelAccuracy;
results.SampleSize = size(X, 1);
results.NumFeatures = numFeatures;

%% Display Results
if verbose
    fprintf('\n=== FEATURE IMPORTANCE RESULTS ===\n');
    fprintf('%-25s', 'Feature');
    for i = 1:numMethods
        fprintf(' | %-15s', methodLabels{i});
    end
    fprintf(' | %-12s | %s\n', 'Combined', 'Rank');
    fprintf('%s\n', repmat('-', 1, 25 + numMethods * 19 + 30));
    
    for i = sortIdx'  % Display in order of importance
        fprintf('%-25s', featureNames{i});
        for j = 1:numMethods
            fprintf(' | %15.4f', scores(i, j));
        end
        fprintf(' | %12.4f | %4d\n', combinedScore(i), combinedRank(i));
    end
    fprintf('\n');
    
    if ~isnan(modelAccuracy)
        if strcmp(problemType, 'classification')
            fprintf('Model Performance: %.2f%% accuracy\n', modelAccuracy * 100);
        else
            fprintf('Model Performance: R² = %.4f\n', modelAccuracy);
        end
    end
end

%% Visualization
if plotResults
    figHandle = figure('Position', [100, 100, 1200, 600], 'Color', 'w');
    
    % Sort features by combined importance
    [~, sortIdx] = sort(combinedScore, 'descend');
    
    % Plot 1: Heatmap of importance scores
    subplot(1, 2, 1);
    imagesc(scoresNorm(sortIdx, :)');
    colormap(jet);
    colorbar;
    set(gca, 'YTick', 1:numMethods, 'YTickLabel', methodLabels);
    set(gca, 'XTick', 1:numFeatures, 'XTickLabel', featureNames(sortIdx), ...
        'XTickLabelRotation', 45);
    title(sprintf('Feature Importance Heatmap\n(%s)', upper(problemType)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Features (sorted by importance)');
    ylabel('Method');
    
    % Plot 2: Bar chart of combined importance
    subplot(1, 2, 2);
    barh(1:numFeatures, combinedScore(sortIdx));
    set(gca, 'YTick', 1:numFeatures, 'YTickLabel', featureNames(sortIdx));
    title(sprintf('Combined Feature Importance\nTarget: %s', targetName), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Importance Score (normalized)');
    ylabel('Features');
    grid on;
    xlim([0, 1.1 * max(combinedScore)]);
    
    % Add value labels on bars
    for i = 1:numFeatures
        text(combinedScore(sortIdx(i)) + 0.02 * max(combinedScore), i, ...
            sprintf('%.3f', combinedScore(sortIdx(i))), ...
            'VerticalAlignment', 'middle', 'FontSize', 8);
    end
    
    sgtitle(sprintf('Feature Importance Analysis - %d samples, %d features', ...
        size(X, 1), numFeatures), 'FontSize', 14, 'FontWeight', 'bold');
else
    figHandle = [];
end

end

%% Helper Function: Calculate Mutual Information
function mi = calculate_mutual_information(x, y)
    % Remove NaN values
    validIdx = ~isnan(x) & ~isnan(y);
    x = x(validIdx);
    y = y(validIdx);
    
    if isempty(x)
        mi = 0;
        return;
    end
    
    % Calculate joint and marginal probabilities
    uniqueX = unique(x);
    uniqueY = unique(y);
    
    % Joint probability
    pxy = zeros(length(uniqueX), length(uniqueY));
    for i = 1:length(uniqueX)
        for j = 1:length(uniqueY)
            pxy(i, j) = sum(x == uniqueX(i) & y == uniqueY(j)) / length(x);
        end
    end
    
    % Marginal probabilities
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    
    % Mutual information
    mi = 0;
    for i = 1:length(uniqueX)
        for j = 1:length(uniqueY)
            if pxy(i, j) > 0
                mi = mi + pxy(i, j) * log2(pxy(i, j) / (px(i) * py(j)));
            end
        end
    end
end
