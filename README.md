# Feature Importance Analyzer for MATLAB

## Overview

Function for quantifying feature importance using multiple machine learning approaches. This tool helps identify which features (predictor variables) are most strongly correlated with a target variable.

## Key Features

- **Automatic Problem Detection**: Automatically identifies whether the problem is classification or regression
- **Multiple Analysis Methods**: 
  - Random Forest feature importance (ensemble-based)
  - ReliefF algorithm (distance-based)
  - Correlation analysis (for regression)
  - Mutual Information (information-theoretic)
- **Flexible Input**: Interactive mode or programmatic specification
- **Robust Results**: Combines multiple methods for reliable feature ranking
- **Visualizations**: Heatmaps and bar charts
- **Comprehensive Output**: Detailed results structure with scores, ranks, and model performance

## Installation

Simply add the `analyze_feature_importance.m` file to your MATLAB path or current working directory.

```matlab
% Add to path
addpath('/path/to/feature_importance_analyzer');

% Or navigate to the directory
cd '/path/to/feature_importance_analyzer'
```

## Quick Start

### Interactive Mode
```matlab
% Load your data as a MATLAB table
load fisheriris
dataTable = array2table([meas, grp2idx(species)], ...
    'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'});

% Run with interactive prompts
results = analyze_feature_importance(dataTable);
```

### Programmatic Mode
```matlab
% Specify columns explicitly
results = analyze_feature_importance(dataTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species');
```

## Function Syntax

```matlab
results = analyze_feature_importance(dataTable)
results = analyze_feature_importance(dataTable, Name, Value)
[results, figHandle] = analyze_feature_importance(...)
```

### Input Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `dataTable` | table | MATLAB table containing your data |

### Optional Name-Value Pairs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FeatureColumns` | `[]` | Cell array or numeric indices of feature columns |
| `TargetColumn` | `[]` | Name or index of target variable |
| `Method` | `'auto'` | Analysis method: 'auto', 'all', 'randomforest', 'relief', 'correlation', 'mutualinfo' |
| `NumTrees` | `100` | Number of trees for Random Forest |
| `Normalize` | `true` | Whether to normalize features |
| `PlotResults` | `true` | Whether to create visualizations |
| `Verbose` | `true` | Whether to display progress messages |

### Output Structure

```matlab
results = 
  struct with fields:
    FeatureNames: {1×4 cell}        % Names of analyzed features
    TargetName: 'Species'            % Name of target variable
    ProblemType: 'classification'    % 'classification' or 'regression'
    Methods: {1×3 cell}              % Methods used
    Scores: [4×3 double]             % Raw importance scores
    Ranks: [4×3 double]              % Feature ranks by method
    NormalizedScores: [4×3 double]   % Normalized scores [0,1]
    CombinedScore: [4×1 double]      % Average normalized score
    CombinedRank: [4×1 double]       % Final ranking
    ModelAccuracy: 0.9733            % Model performance metric
    SampleSize: 150                  % Number of observations
    NumFeatures: 4                   % Number of features analyzed
```

## Analysis Methods Explained

### 1. Random Forest Feature Importance
Uses ensemble of decision trees to measure how much each feature decreases prediction error. Most robust and widely used method.

**Best for**: General-purpose feature importance, both classification and regression

### 2. ReliefF Algorithm
Evaluates features based on how well they distinguish between similar instances. Effective for detecting feature interactions.

**Best for**: Features with complex relationships, noisy data

### 3. Correlation Analysis (Regression only)
Measures linear correlation between each feature and target variable.

**Best for**: Understanding linear relationships, quick preliminary analysis

### 4. Mutual Information
Measures how much information a feature provides about the target variable. Captures non-linear relationships.

**Best for**: Non-linear relationships, categorical features

## Usage Examples

### Example 1: Classification Problem
```matlab
% Load iris dataset
load fisheriris
irisTable = array2table([meas, grp2idx(species)], ...
    'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'});

% Analyze feature importance
results = analyze_feature_importance(irisTable, ...
    'FeatureColumns', 1:4, ...
    'TargetColumn', 'Species', ...
    'Method', 'all');

% Display top features
[~, idx] = sort(results.CombinedScore, 'descend');
fprintf('Most important features:\n');
for i = 1:3
    fprintf('%d. %s (Score: %.4f)\n', i, ...
        results.FeatureNames{idx(i)}, results.CombinedScore(idx(i)));
end
```

### Example 2: Regression Problem
```matlab
% Load car data
load carsmall
carTable = table(Acceleration, Cylinders, Displacement, Horsepower, Weight, MPG);
carTable = rmmissing(carTable);

% Analyze which features predict MPG
results = analyze_feature_importance(carTable, ...
    'FeatureColumns', 1:5, ...
    'TargetColumn', 'MPG', ...
    'NumTrees', 150);

% Model performance
fprintf('Model R² Score: %.4f\n', results.ModelAccuracy);
```

### Example 3: High-Dimensional Feature Selection
```matlab
% Generate synthetic data with 50 features, only 5 important
rng(42);
nSamples = 200;
nFeatures = 50;
X = randn(nSamples, nFeatures);
y = 2*X(:,5) + 3*X(:,12) - X(:,25) + 1.5*X(:,37) + 4*X(:,48) + randn(nSamples,1);

% Create table
featureNames = compose('F%02d', 1:nFeatures);
dataTable = array2table([X, y], 'VariableNames', [featureNames, {'Target'}]);

% Find important features
results = analyze_feature_importance(dataTable, ...
    'FeatureColumns', 1:nFeatures, ...
    'TargetColumn', 'Target', ...
    'Method', 'all');

% Get top 10 features
[~, topIdx] = maxk(results.CombinedScore, 10);
fprintf('Top 10 features: %s\n', strjoin(results.FeatureNames(topIdx), ', '));
```

## Interpreting Results

### Understanding Scores
- **Higher score = More important feature**
- Scores are method-specific and may not be directly comparable
- Use `NormalizedScores` or `CombinedScore` for cross-method comparison

### Combined Rankings
The `CombinedScore` is the average of normalized scores across all methods, providing a robust consensus ranking:
- Values range from 0 to 1
- Features with score > 0.7 are typically highly important
- Features with score < 0.3 may be candidates for removal

### Model Accuracy
- **Classification**: OOB (Out-of-Bag) accuracy percentage
- **Regression**: R² score (coefficient of determination)
- Values closer to 1.0 indicate better model fit

## Best Practices

### 1. Data Preparation
```matlab
% Remove missing data
dataTable = rmmissing(dataTable);

% Check for constant features
variances = varfun(@var, dataTable(:, featureColumns));
constantFeatures = variances{1,:} == 0;
if any(constantFeatures)
    warning('Some features have zero variance');
end
```

### 2. Feature Scaling
The function automatically normalizes features by default. For raw comparisons:
```matlab
results = analyze_feature_importance(dataTable, ...
    'Normalize', false);
```

### 3. Method Selection
- For quick analysis: Use `'Method', 'auto'` (default)
- For comprehensive analysis: Use `'Method', 'all'`
- For specific needs: Choose individual method

### 4. Sample Size Considerations
- Minimum recommended: 50 samples
- For reliable results: 200+ samples
- High-dimensional data: samples >> features

### 5. Multiple Runs
For critical applications, run analysis multiple times with different random seeds:
```matlab
nRuns = 10;
allScores = zeros(nFeatures, nRuns);
for i = 1:nRuns
    rng(i);
    results = analyze_feature_importance(dataTable, ...
        'FeatureColumns', featureCols, ...
        'TargetColumn', targetCol, ...
        'Verbose', false);
    allScores(:, i) = results.CombinedScore;
end
avgScore = mean(allScores, 2);
stdScore = std(allScores, [], 2);
```

## Advanced Usage

### Custom Visualization
```matlab
% Run without built-in plots
results = analyze_feature_importance(dataTable, ...
    'PlotResults', false);

% Create custom visualization
figure;
barh(results.CombinedScore);
set(gca, 'YTick', 1:length(results.FeatureNames), ...
    'YTickLabel', results.FeatureNames);
title('Custom Feature Importance Plot');
xlabel('Importance Score');
```

### Exporting Results
```matlab
% Create results table
resultsTable = table(results.FeatureNames', ...
    results.CombinedScore, ...
    results.CombinedRank, ...
    'VariableNames', {'Feature', 'Score', 'Rank'});

% Sort by importance
resultsTable = sortrows(resultsTable, 'Rank');

% Save to CSV
writetable(resultsTable, 'feature_importance.csv');

% Save to Excel
writetable(resultsTable, 'feature_importance.xlsx');
```

### Integration with Model Building
```matlab
% Select top N features
topN = 5;
[~, topIdx] = maxk(results.CombinedScore, topN);
topFeatures = results.FeatureNames(topIdx);

% Build model with selected features
X = dataTable{:, topFeatures};
y = dataTable{:, results.TargetName};

% Train final model
finalModel = fitcensemble(X, y, 'Method', 'Bag', ...
    'NumLearningCycles', 100);
```

## Troubleshooting

### Issue: "Feature column not found"
**Solution**: Check column names are exact matches (case-sensitive)
```matlab
dataTable.Properties.VariableNames  % View all column names
```

### Issue: Low model accuracy
**Possible causes**:
- Insufficient data
- Irrelevant features
- Non-linear relationships not captured
- Data quality issues

**Solutions**:
- Increase sample size
- Try feature engineering
- Check for outliers
- Verify data preprocessing

### Issue: Different methods give conflicting rankings
**This is normal!** Different methods measure different aspects:
- Use `CombinedScore` for consensus
- Consider domain knowledge
- Look for features ranked highly by multiple methods

### Issue: Memory errors with large datasets
**Solutions**:
```matlab
% Reduce number of trees
results = analyze_feature_importance(dataTable, 'NumTrees', 50);

% Analyze feature subsets
nFeatures = width(dataTable) - 1;
chunkSize = 20;
for i = 1:chunkSize:nFeatures
    endIdx = min(i+chunkSize-1, nFeatures);
    featureChunk = i:endIdx;
    results = analyze_feature_importance(dataTable, ...
        'FeatureColumns', featureChunk);
end
```

## Performance Considerations

| Dataset Size | Features | Approximate Time |
|--------------|----------|------------------|
| Small (100 samples, 5 features) | All methods | < 5 seconds |
| Medium (1000 samples, 20 features) | All methods | 10-30 seconds |
| Large (10000 samples, 50 features) | All methods | 1-2 minutes |
| Very Large (50000+ samples, 100+ features) | Random Forest only | 2-5 minutes |

**Tips for faster execution**:
- Reduce `NumTrees` for preliminary analysis
- Use `'Method', 'randomforest'` for large datasets
- Disable plotting with `'PlotResults', false`
- Set `'Verbose', false` to reduce overhead

## Comparison with Other Tools

| Tool | Pros | Cons |
|------|------|------|
| **This Function** | Multiple methods, automatic detection, comprehensive output | Requires MATLAB with Statistics toolbox |
| **Python scikit-learn** | More algorithms available | Requires Python environment |
| **R caret** | Extensive preprocessing | Different syntax, R-specific |
| **MATLAB fitcensemble** | Native MATLAB | Single method only |

## Requirements

- MATLAB R2019b or later (recommended: R2021a+)
- Statistics and Machine Learning Toolbox
- Minimum 4GB RAM (8GB+ recommended for large datasets)

## License

MIT License - Free for academic and commercial use

## Changelog

### Version 1.0 (2025-01-15)
- Initial release
- Support for classification and regression
- Four feature importance methods
- Automatic problem detection
- Comprehensive visualization

## Acknowledgments

This tool was developed based on best practices from:
- Breiman, L. (2001). "Random Forests"
- Kononenko, I. (1994). "Estimating attributes: Analysis and extensions of RELIEF"
- Cover, T.M. and Thomas, J.A. (2006). "Elements of Information Theory"

## FAQ

**Q: Can I use this with time series data?**
A: Yes, but be aware of temporal dependencies. Consider using lagged features.

**Q: What if I have categorical features?**
A: Convert them to dummy variables first using `dummyvar()` or one-hot encoding.

**Q: How do I handle imbalanced classes?**
A: The Random Forest implementation handles this naturally. Consider using class weights if needed.

**Q: Can I use this for deep learning feature selection?**
A: Yes, use the results to select features before training neural networks.

**Q: What's the difference between feature importance and feature selection?**
A: Importance quantifies relevance; selection is the process of choosing features based on importance or other criteria.
