%% QUICK START GUIDE - Feature Importance Analyzer
% Get started with feature importance analysis in 5 minutes

%% STEP 1: Prepare Your Data
% Your data must be in a MATLAB table format

% Example: Load sample data
load fisheriris
dataTable = array2table([meas, grp2idx(species)], ...
    'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'});

% Or import your own data:
% dataTable = readtable('your_data.csv');

%% STEP 2: Run the Analysis (Easiest Way)
% Just pass your table - the function will prompt you for column selection
results = analyze_feature_importance(dataTable);

%% STEP 3: Run with Explicit Column Selection
% Specify which columns are features and which is the target
results = analyze_feature_importance(dataTable, ...
    'FeatureColumns', 1:4, ...          % Columns 1-4 are features
    'TargetColumn', 'Species');         % 'Species' is the target

%% STEP 4: View Results
% The function automatically displays results and creates plots

% Access specific results:
fprintf('Most important feature: %s\n', results.FeatureNames{1});
fprintf('Model accuracy: %.2f%%\n', results.ModelAccuracy * 100);

% See all feature rankings:
disp(table(results.FeatureNames', results.CombinedScore, results.CombinedRank, ...
    'VariableNames', {'Feature', 'Score', 'Rank'}));

%% STEP 5: Use Different Methods
% Try different analysis methods for comprehensive insights

% Use all methods (recommended for important decisions)
results_all = analyze_feature_importance(dataTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species', ...
    'Method', 'all');

% Use specific method
results_rf = analyze_feature_importance(dataTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species', ...
    'Method', 'randomforest', ...
    'NumTrees', 200);

%% BONUS: Select Top Features for Modeling
% Use the results to select the most important features

% Get top N features
topN = 3;
[~, topIdx] = maxk(results.CombinedScore, topN);
topFeatures = results.FeatureNames(topIdx);

fprintf('\nTop %d features to use in your model:\n', topN);
disp(topFeatures);

% Extract data for top features only
X_selected = dataTable{:, topFeatures};
y = dataTable{:, results.TargetName};

% Now build your model with selected features
% model = fitcensemble(X_selected, y);

%% COMMON PARAMETERS CHEAT SHEET
%
% NAME              DEFAULT     OPTIONS
% ----------------  ----------  ------------------------------------------
% FeatureColumns    []          Column indices or names: 1:5 or {'A','B'}
% TargetColumn      []          Column index or name: 6 or 'Target'
% Method            'auto'      'auto', 'all', 'randomforest', 'relief'
% NumTrees          100         Any positive integer (50-200 typical)
% Normalize         true        true or false
% PlotResults       true        true or false
% Verbose           true        true or false
%
%% EXAMPLE WORKFLOWS

% FOR REGRESSION:
load carsmall
carTable = table(Acceleration, Cylinders, Displacement, Horsepower, Weight, MPG);
carTable = rmmissing(carTable);
results = analyze_feature_importance(carTable, ...
    'FeatureColumns', 1:5, 'TargetColumn', 'MPG');

% FOR CLASSIFICATION:
load fisheriris
irisTable = array2table([meas, grp2idx(species)], ...
    'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'});
results = analyze_feature_importance(irisTable, ...
    'FeatureColumns', 1:4, 'TargetColumn', 'Species');

% FOR HIGH-DIMENSIONAL DATA (many features):
% results = analyze_feature_importance(dataTable, ...
%     'FeatureColumns', 1:100, 'TargetColumn', 'Target', ...
%     'Method', 'randomforest', 'NumTrees', 150);

%% INTERPRETING OUTPUT

% results.CombinedScore  -> Higher = more important (0 to ~1)
% results.CombinedRank   -> 1 = most important
% results.ModelAccuracy  -> How well the model fits (classification: %, regression: R²)
% 
% SCORE INTERPRETATION:
%   > 0.7  -> Highly important, definitely keep
%   0.4-0.7 -> Moderately important, consider keeping
%   < 0.4  -> Less important, may consider removing

%% TIPS
% 
% 1. Always remove missing data first: dataTable = rmmissing(dataTable);
% 2. For critical analysis, use 'Method', 'all'
% 3. Try different NumTrees values (50, 100, 200) to check stability
% 4. Features ranked differently by methods? Use CombinedScore!
% 5. Save your results: save('my_results.mat', 'results');

%% TROUBLESHOOTING
%
% ERROR: "Column not found"
%   -> Check column names: dataTable.Properties.VariableNames
%   -> Names are case-sensitive!
%
% Low accuracy?
%   -> Need more data
%   -> Features might not be predictive
%   -> Try feature engineering
%
% Taking too long?
%   -> Reduce NumTrees: 'NumTrees', 50
%   -> Use fewer features
%   -> Try 'Method', 'relief' (faster than Random Forest)

%% NEXT STEPS
%
% 1. Run the full examples: open('example_usage.m')
% 2. Read the documentation: open('README.md')
% 3. Try with your own data!
% 4. Use selected features to build your final model

fprintf('\n✓ Quick Start Guide Complete!\n');
fprintf('Ready to analyze your data? Load your table and run:\n');
fprintf('  results = analyze_feature_importance(yourTable);\n\n');
