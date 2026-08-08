clear;
close all;
global Configuration;

% Compare WD with MMD variants inside the same dual-metric selection framework.
% This script is independent from demo.m and does not affect the main workflow.

%sourceDataset = "wheatA1_20048";
%targetDataset = "wheatA3_48200";
sourceDataset = "nirshootout1"; 
targetDataset = "nirshootout2"; 
task = 'proposed';
fold = 4;
LV = 10;

tStart = tic;
Configuration = [];

fprintf('\n[Metric Stability] Source dataset: %s\n', sourceDataset);
fprintf('[Metric Stability] Target dataset: %s\n', targetDataset);

source_data = load(sourceDataset);
target_data = load(targetDataset);

X_source = source_data.X;
y_source = source_data.y;
X_target = target_data.X;
X_target_test = target_data.Xtest;
y_target_test = target_data.ytest;

buildConfigration(task, fold, LV);
Configuration.Xtarget = X_target;
Configuration.Xtarget_RAW = X_target;
Configuration.XtrainData = X_source;
Configuration.ytrainData = y_source;
Configuration.Xtarget_test = X_target_test;
Configuration.ytarget_test = y_target_test;

fprintf('\n[Metric Stability] Building one shared GA candidate pool...\n');
[~, optimization_results] = preprocessCombos_domain(X_source, y_source);
details = optimization_results.details;
n_candidates = numel(details);

fprintf('[Metric Stability] Evaluating %d candidates with WD and MMD metrics...\n', n_candidates);

rmsecv = [details.rmsecv]';
wd_values = [details.wass_dist]';
mmd_rbf = zeros(n_candidates, 1);
mmd_laplacian = zeros(n_candidates, 1);
mmd_linear = zeros(n_candidates, 1);

for i = 1:n_candidates
    combo = details(i).combo;
    lv = details(i).lv;
    [yhat_source, yhat_target] = predictSourceAndTarget( ...
        X_source, y_source, X_target, combo, lv);

    % RBF bandwidth follows Umprecht et al.: median pairwise distance of
    % source-domain predictions.
    gamma = sourcePredictionMedianDistance(yhat_source);
    mmd_rbf(i) = calcMmdSquared(yhat_source, yhat_target, 'rbf', gamma);
    mmd_laplacian(i) = calcMmdSquared(yhat_source, yhat_target, 'laplacian', gamma);
    mmd_linear(i) = calcMmdSquared(yhat_source, yhat_target, 'linear', gamma);

    if mod(i, 50) == 0
        fprintf('.');
    end
end
fprintf('\n');

metric_specs = struct( ...
    'name', {'WD', 'MMD-RBF', 'MMD-Laplacian', 'MMD-Linear'}, ...
    'values', {wd_values, mmd_rbf, mmd_laplacian, mmd_linear});

rows = struct( ...
    'dataset', {}, ...
    'metric', {}, ...
    'selected_idx', {}, ...
    'lv', {}, ...
    'rmsecv', {}, ...
    'discrepancy', {}, ...
    'selection_score', {}, ...
    'target_rmsep', {}, ...
    'target_r2', {}, ...
    'target_rpd', {}, ...
    'combo', {});

for m = 1:numel(metric_specs)
    metric_name = metric_specs(m).name;
    metric_values = metric_specs(m).values;

    score = normalizeMinMax(rmsecv).^2 + normalizeMinMax(metric_values).^2;
    [~, best_idx] = min(score);

    combo = details(best_idx).combo;
    lv = details(best_idx).lv;
    [~, yhat_target_test] = predictSourceAndTarget( ...
        X_source, y_source, X_target_test, combo, lv);
    [target_rmsep, target_r2, target_rpd] = regressionMetrics(y_target_test, yhat_target_test);

    rows(end+1).dataset = char(targetDataset); %#ok<SAGROW>
    rows(end).metric = metric_name;
    rows(end).selected_idx = best_idx;
    rows(end).lv = lv;
    rows(end).rmsecv = rmsecv(best_idx);
    rows(end).discrepancy = metric_values(best_idx);
    rows(end).selection_score = score(best_idx);
    rows(end).target_rmsep = target_rmsep;
    rows(end).target_r2 = target_r2;
    rows(end).target_rpd = target_rpd;
    rows(end).combo = getComboName(combo);

    fprintf(['[Metric Stability] %s | idx=%d | LV=%d | RMSECV=%.4f | ' ...
        'Discrepancy=%.6f | RMSEP=%.4f | R2=%.4f | RPD=%.4f | %s\n'], ...
        metric_name, best_idx, lv, rmsecv(best_idx), metric_values(best_idx), ...
        target_rmsep, target_r2, target_rpd, rows(end).combo);
end

result_table = struct2table(rows);

fprintf('\n[Metric Stability] Summary table:\n');
disp(result_table);
fprintf('[Metric Stability] Total elapsed time: %.2f seconds\n', toc(tStart));

function [yhat_source, yhat_other] = predictSourceAndTarget(X_source, y_source, X_other, combo, lv)
    global Configuration;
    Xs = X_source;
    Xo = X_other;
    for k = 1:length(combo)
        [Xs, Xo] = Configuration.Backbone{k}{combo(k)}(Xs, Xo);
    end
    [~,~,~,~,BETA] = plsregress(Xs, y_source, lv);
    yhat_source = [ones(size(Xs, 1), 1), Xs] * BETA;
    yhat_other = [ones(size(Xo, 1), 1), Xo] * BETA;
end

function gamma = sourcePredictionMedianDistance(yhat_source)
    ys = yhat_source(:);
    dist_mat = abs(ys - ys');
    gamma = median(dist_mat(triu(true(size(dist_mat)))));
    if ~isfinite(gamma) || gamma <= 0
        gamma = eps;
    end
end

function mmd2 = calcMmdSquared(y_source, y_target, kernel_name, gamma)
    ys = y_source(:);
    yt = y_target(:);
    ns = numel(ys);
    nt = numel(yt);

    Kss = kernelMatrix(ys, ys, kernel_name, gamma);
    Ktt = kernelMatrix(yt, yt, kernel_name, gamma);
    Kst = kernelMatrix(ys, yt, kernel_name, gamma);

    source_term = (sum(Kss(:)) - trace(Kss)) / (ns * (ns - 1));
    target_term = (sum(Ktt(:)) - trace(Ktt)) / (nt * (nt - 1));
    cross_term = 2 * mean(Kst(:));
    mmd2 = max(source_term + target_term - cross_term, 0);
end

function K = kernelMatrix(a, b, kernel_name, gamma)
    A = a(:);
    B = b(:)';
    switch lower(kernel_name)
        case 'rbf'
            K = exp(-((A - B).^2) ./ gamma);
        case 'laplacian'
            K = exp(-abs(A - B) ./ gamma);
        case 'linear'
            K = A * B;
        otherwise
            error('Unknown kernel: %s', kernel_name);
    end
end

function x_norm = normalizeMinMax(x)
    x = x(:);
    xmin = min(x);
    xmax = max(x);
    if xmax == xmin
        x_norm = zeros(size(x));
    else
        x_norm = (x - xmin) ./ (xmax - xmin);
    end
end

function [rmsep, r2, rpd] = regressionMetrics(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    rmsep = sqrt(mean((y_pred - y_true).^2));
    sst = sum((y_true - mean(y_true)).^2);
    sse = sum((y_pred - y_true).^2);
    r2 = 1 - sse / sst;
    rpd = std(y_true) / rmsep;
end

function name = getComboName(combo)
    global Configuration;
    parts = strings(1, length(combo));
    for k = 1:length(combo)
        func_str = func2str(Configuration.Backbone{k}{combo(k)});
        func_str = regexprep(func_str, '^(smothing_|scatter_|baseline_|scaling_|centering_)', '');
        parts(k) = string(func_str);
    end
    name = char(strjoin(parts, ' + '));
end
