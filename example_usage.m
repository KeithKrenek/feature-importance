%% FEATURE IMPORTANCE ANALYZER - USAGE EXAMPLES
% This script demonstrates various ways to use the analyze_feature_importance function
% for both classification and regression problems.

clear; close all; clc;

%% EXAMPLE 1: Classification - Fisher's Iris Dataset (Interactive Mode)
fprintf('=============================================================\n');
fprintf('EXAMPLE 1: Classification - Interactive Mode\n');
fprintf('=============================================================\n\n');

% Load and prepare iris dataset
load fisheriris
irisTable = array2table([meas, grp2idx(species)], ...
    'VariableNames', {'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'});

fprintf('Using Fisher''s Iris dataset for species classification.\n\n');

% Run analysis with interactive column selection
% Uncomment the following line to try interactive mode:
% results1 = analyze_feature_importance(irisTable);

% Or use programmatic mode:
results1 = analyze_feature_importance(irisTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species', ...
    'Method', 'all', ...
    'PlotResults', true);

fprintf('\nPress any key to continue to Example 2...\n');
pause;

%% EXAMPLE 2: Regression - Car Fuel Efficiency
fprintf('\n=============================================================\n');
fprintf('EXAMPLE 2: Regression - Car Fuel Efficiency\n');
fprintf('=============================================================\n\n');

% Load and prepare car dataset
load carsmall

% Create table (remove rows with missing data)
carTable = table(Acceleration, Cylinders, Displacement, Horsepower, Weight, MPG);
carTable = rmmissing(carTable);

fprintf('Analyzing which car features predict fuel efficiency (MPG).\n\n');

% Run analysis
results2 = analyze_feature_importance(carTable, ...
    'FeatureColumns', {'Acceleration', 'Cylinders', 'Displacement', 'Horsepower', 'Weight'}, ...
    'TargetColumn', 'MPG', ...
    'Method', 'auto', ...
    'NumTrees', 150, ...
    'PlotResults', true);

fprintf('\nPress any key to continue to Example 3...\n');
pause;

%% EXAMPLE 3: Custom Dataset - Medical Diagnosis Classification
fprintf('\n=============================================================\n');
fprintf('EXAMPLE 3: Custom Dataset - Medical Diagnosis\n');
fprintf('=============================================================\n\n');

% Generate synthetic medical dataset
rng(42); % For reproducibility
n = 500;

% Features: Age, BMI, Blood Pressure, Cholesterol, Blood Sugar
age = 20 + 60 * rand(n, 1);
bmi = 18 + 20 * rand(n, 1);
bloodPressure = 90 + 50 * rand(n, 1);
cholesterol = 150 + 100 * rand(n, 1);
bloodSugar = 70 + 80 * rand(n, 1);

% Target: Disease risk (High/Medium/Low) - synthetic relationship
riskScore = 0.3 * (age - 40) + 0.4 * (bmi - 25) + 0.2 * (bloodPressure - 120) + ...
    0.1 * (cholesterol - 200) + 0.2 * (bloodSugar - 100) + 10 * randn(n, 1);

diagnosis = categorical(zeros(n, 1), [1 2 3], {'Low', 'Medium', 'High'});
diagnosis(riskScore < -5) = 'Low';
diagnosis(riskScore >= -5 & riskScore < 5) = 'Medium';
diagnosis(riskScore >= 5) = 'High';

% Create table
medicalTable = table(age, bmi, bloodPressure, cholesterol, bloodSugar, diagnosis, ...
    'VariableNames', {'Age', 'BMI', 'BloodPressure', 'Cholesterol', 'BloodSugar', 'RiskLevel'});

fprintf('Analyzing which medical factors predict disease risk level.\n\n');

% Run analysis with all methods
results3 = analyze_feature_importance(medicalTable, ...
    'FeatureColumns', 1:5, ...
    'TargetColumn', 'RiskLevel', ...
    'Method', 'all', ...
    'NumTrees', 200, ...
    'PlotResults', true);

fprintf('\nPress any key to continue to Example 4...\n');
pause;

%% EXAMPLE 4: High-Dimensional Regression with Feature Selection
fprintf('\n=============================================================\n');
fprintf('EXAMPLE 4: High-Dimensional Data - Feature Selection\n');
fprintf('=============================================================\n\n');

% Generate high-dimensional dataset
rng(123);
nSamples = 300;
nFeatures = 20;

% Most features are noise, only 5 are truly important
X_data = randn(nSamples, nFeatures);

% True relationship: y depends on features 3, 7, 11, 15, 18
y_true = 5 * X_data(:, 3) + 3 * X_data(:, 7) - 4 * X_data(:, 11) + ...
    2 * X_data(:, 15) + 6 * X_data(:, 18) + randn(nSamples, 1);

% Create feature names
featureNames = cell(1, nFeatures);
for i = 1:nFeatures
    featureNames{i} = sprintf('Feature_%02d', i);
end

% Create table
highDimTable = array2table([X_data, y_true], ...
    'VariableNames', [featureNames, {'Target'}]);

fprintf('Dataset with %d features where only 5 are truly predictive.\n', nFeatures);
fprintf('True important features: 3, 7, 11, 15, 18\n\n');

% Run analysis
results4 = analyze_feature_importance(highDimTable, ...
    'FeatureColumns', 1:nFeatures, ...
    'TargetColumn', 'Target', ...
    'Method', 'all', ...
    'NumTrees', 100, ...
    'PlotResults', true);

% Display top 10 features
fprintf('\nTop 10 Most Important Features:\n');
[~, sortIdx] = sort(results4.CombinedScore, 'descend');
for i = 1:min(10, length(sortIdx))
    fprintf('%2d. %s (Score: %.4f)\n', i, ...
        results4.FeatureNames{sortIdx(i)}, results4.CombinedScore(sortIdx(i)));
end

fprintf('\nPress any key to continue to Advanced Analysis...\n');
pause;

%% EXAMPLE 5: Advanced - Comparing Multiple Methods
fprintf('\n=============================================================\n');
fprintf('EXAMPLE 5: Advanced Analysis - Method Comparison\n');
fprintf('=============================================================\n\n');

% Using the iris dataset again for detailed method comparison
fprintf('Comparing different feature importance methods on Iris dataset:\n\n');

% Test individual methods
methods = {'randomforest', 'relief', 'mutualinfo'};
allResults = cell(1, length(methods));

for i = 1:length(methods)
    fprintf('Running %s method...\n', methods{i});
    allResults{i} = analyze_feature_importance(irisTable, ...
        'FeatureColumns', 1:4, ...
        'TargetColumn', 'Species', ...
        'Method', methods{i}, ...
        'PlotResults', false, ...
        'Verbose', false);
end

% Create comparison figure
figure('Position', [100, 100, 1000, 600], 'Color', 'w');

for i = 1:length(methods)
    subplot(2, 2, i);
    scores = allResults{i}.Scores;
    [sortedScores, sortIdx] = sort(scores, 'descend');
    
    bar(sortedScores);
    set(gca, 'XTickLabel', allResults{i}.FeatureNames(sortIdx), ...
        'XTickLabelRotation', 45);
    title(sprintf('%s Method', upper(methods{i})), 'FontWeight', 'bold');
    ylabel('Importance Score');
    xlabel('Features');
    grid on;
end

% Combined results
subplot(2, 2, 4);
resultsAll = analyze_feature_importance(irisTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species', ...
    'Method', 'all', ...
    'PlotResults', false, ...
    'Verbose', false);

[sortedScores, sortIdx] = sort(resultsAll.CombinedScore, 'descend');
bar(sortedScores);
set(gca, 'XTickLabel', resultsAll.FeatureNames(sortIdx), ...
    'XTickLabelRotation', 45);
title('COMBINED (All Methods)', 'FontWeight', 'bold');
ylabel('Normalized Importance');
xlabel('Features');
grid on;

sgtitle('Method Comparison: Feature Importance Analysis', ...
    'FontSize', 14, 'FontWeight', 'bold');

%% EXAMPLE 6: Exporting Results
fprintf('\n=============================================================\n');
fprintf('EXAMPLE 6: Exporting Results to Files\n');
fprintf('=============================================================\n\n');

% Create a comprehensive results table
resultsTable = table();
resultsTable.Feature = results1.FeatureNames(:);
resultsTable.CombinedScore = results1.CombinedScore;
resultsTable.Rank = results1.CombinedRank;

% Add individual method scores
for i = 1:length(results1.Methods)
    resultsTable.(results1.Methods{i}) = results1.Scores(:, i);
end

% Display table
disp(resultsTable);

% Save to CSV (uncomment to save)
% writetable(resultsTable, 'feature_importance_results.csv');
% fprintf('\nResults saved to: feature_importance_results.csv\n');

% Save workspace (uncomment to save)
% save('feature_importance_analysis.mat', 'results1', 'results2', 'results3', 'results4');
% fprintf('Workspace saved to: feature_importance_analysis.mat\n');

%% Summary
fprintf('\n=============================================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('=============================================================\n\n');
fprintf('This script demonstrated:\n');
fprintf('  1. Classification analysis (Iris dataset)\n');
fprintf('  2. Regression analysis (Car MPG prediction)\n');
fprintf('  3. Custom medical diagnosis dataset\n');
fprintf('  4. High-dimensional feature selection\n');
fprintf('  5. Method comparison across algorithms\n');
fprintf('  6. Exporting results to files\n\n');

fprintf('Key Takeaways:\n');
fprintf('  • Different methods may rank features differently\n');
fprintf('  • Combined score provides robust feature ranking\n');
fprintf('  • Random Forest importance is generally most reliable\n');
fprintf('  • Use multiple methods for comprehensive analysis\n');
fprintf('  • Normalize data for best results\n\n');

fprintf('Next Steps:\n');
fprintf('  • Try with your own dataset\n');
fprintf('  • Experiment with different numbers of trees\n');
fprintf('  • Use selected features for model building\n');
fprintf('  • Iterate based on domain knowledge\n\n');
