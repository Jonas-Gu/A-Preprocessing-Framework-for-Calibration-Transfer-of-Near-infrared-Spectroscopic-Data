clear;
close all;
global Configuration;

%sourceDataset = "nirshootout1";
%targetDataset = "nirshootout2";
sourceDataset = "wheatA1_20048"; 
targetDataset = "wheatA3_48200"; 
task = 'proposed';
fold = 4;
LV = 10;

Configuration = [];
source_data = load(sourceDataset);
target_data = load(targetDataset);
buildConfigration(task, fold, LV);

Configuration.Xtarget = target_data.X;
Configuration.Xtarget_RAW = target_data.X;
Configuration.XtrainData = source_data.X;
Configuration.ytrainData = source_data.y;
Configuration.Xtarget_test = target_data.Xtest;
Configuration.ytarget_test = target_data.ytest;

lv_values = 6:LV;

fprintf('\n========================================\n');
fprintf('[Benchmark] GA\n');
fprintf('========================================\n');
ga_summary = runGaBenchmark(source_data, target_data);

fprintf('\n========================================\n');
fprintf('[Benchmark] Grid search\n');
fprintf('========================================\n');
[grid_summary, candidate_table] = runGridSearchBenchmark(source_data, target_data, lv_values);

comparison_table = buildComparisonTable(ga_summary, grid_summary);
grid_selection_table = buildGridSelectionTable(grid_summary);

plotSearchComparison(comparison_table);

disp(comparison_table);

function summary = runGaBenchmark(source_data, target_data)
    t_start = tic;
    [best_info, optimization_results] = preprocessCombos_domain(source_data.X, source_data.y);
    elapsed_seconds = toc(t_start);

    combo = best_info.x1;
    lv = best_info.LV;
    [target_rmsep, target_r2, target_rpd] = evaluateTargetPerformance( ...
        source_data.X, source_data.y, target_data.Xtest, target_data.ytest, combo, lv);

    candidate_count = numel(optimization_results.details);
    seconds_per_model = elapsed_seconds / max(candidate_count, 1);

    selection_score = NaN;
    if isfield(optimization_results, 'proposed_score')
        selection_score = optimization_results.proposed_score(optimization_results.best_idx);
    end

    summary.method = "GA";
    summary.candidate_count = candidate_count;
    summary.lv = lv;
    summary.combo = combo;
    summary.combo_name = string(getComboName(combo));
    summary.rmsecv = best_info.rmsecv;
    summary.wasserstein = best_info.wass_dist;
    summary.selection_score = selection_score;
    summary.target_rmsep = target_rmsep;
    summary.target_r2 = target_r2;
    summary.target_rpd = target_rpd;
    summary.elapsed_seconds = elapsed_seconds;
    summary.seconds_per_model = seconds_per_model;

    fprintf('\n[GA] Candidate models retained: %d\n', candidate_count);
    fprintf('[GA] Elapsed time: %.2f seconds\n', elapsed_seconds);
    fprintf('[GA] RMSECV: %.4f | WD: %.4f | RMSEP: %.4f | R2: %.4f | RPD: %.4f\n', ...
        summary.rmsecv, summary.wasserstein, target_rmsep, target_r2, target_rpd);
    fprintf('[GA] %s\n', char(summary.combo_name));
end

function [summary, candidate_table] = runGridSearchBenchmark(source_data, target_data, lv_values)
    global Configuration;

    combo_matrix = buildComboMatrix();
    total_models = size(combo_matrix, 1) * numel(lv_values);
    rows(total_models, 1) = struct( ...
        'candidate_index', [], ...
        'lv', [], ...
        'smoothing_idx', [], ...
        'baseline_idx', [], ...
        'scatter_idx', [], ...
        'centering_idx', [], ...
        'scaling_idx', [], ...
        'rmsecv', [], ...
        'wasserstein', [], ...
        'proposed_score', NaN, ...
        'combo', "");

    fprintf('\n[Grid Search] Preprocessing combinations: %d\n', size(combo_matrix, 1));
    fprintf('[Grid Search] LV values: %s\n', strjoin(string(lv_values), ', '));
    fprintf('[Grid Search] Total preprocessing-LV models: %d\n', total_models);

    t_start = tic;
    row_idx = 0;
    for i = 1:numel(lv_values)
        lv = lv_values(i);
        Configuration.LVs = lv;
        fprintf('\n[Grid Search] Evaluating LV=%d\n', lv);

        for j = 1:size(combo_matrix, 1)
            combo = combo_matrix(j, :);
            row_idx = row_idx + 1;

            rmsecv = crossvalidate(source_data.X, source_data.y, combo, ...
                1:size(source_data.X, 2), 0);
            wasserstein = evaluateWasserstein(source_data.X, source_data.y, ...
                target_data.X, combo, lv);

            rows(row_idx).candidate_index = row_idx;
            rows(row_idx).lv = lv;
            rows(row_idx).smoothing_idx = combo(1);
            rows(row_idx).baseline_idx = combo(2);
            rows(row_idx).scatter_idx = combo(3);
            rows(row_idx).centering_idx = combo(4);
            rows(row_idx).scaling_idx = combo(5);
            rows(row_idx).rmsecv = rmsecv;
            rows(row_idx).wasserstein = wasserstein;
            rows(row_idx).combo = string(getComboName(combo));

            if mod(row_idx, 50) == 0
                fprintf('.');
            end
        end
        fprintf('\n');
    end

    candidate_table = struct2table(rows);
    candidate_table.proposed_score = normalizeMinMax(candidate_table.rmsecv).^2 + ...
        normalizeMinMax(candidate_table.wasserstein).^2;

    [~, best_idx] = min(candidate_table.proposed_score);
    best = candidate_table(best_idx, :);
    best_combo = [best.smoothing_idx, best.baseline_idx, best.scatter_idx, ...
        best.centering_idx, best.scaling_idx];

    [target_rmsep, target_r2, target_rpd] = evaluateTargetPerformance( ...
        source_data.X, source_data.y, target_data.Xtest, target_data.ytest, ...
        best_combo, best.lv);

    elapsed_seconds = toc(t_start);
    seconds_per_model = elapsed_seconds / total_models;

    summary.method = "Grid search";
    summary.candidate_index = best.candidate_index;
    summary.candidate_count = total_models;
    summary.lv = best.lv;
    summary.combo = best_combo;
    summary.combo_name = string(best.combo);
    summary.rmsecv = best.rmsecv;
    summary.wasserstein = best.wasserstein;
    summary.selection_score = best.proposed_score;
    summary.target_rmsep = target_rmsep;
    summary.target_r2 = target_r2;
    summary.target_rpd = target_rpd;
    summary.elapsed_seconds = elapsed_seconds;
    summary.seconds_per_model = seconds_per_model;

    fprintf('\n[Grid Search] Candidate models evaluated: %d\n', total_models);
    fprintf('[Grid Search] Elapsed time: %.2f seconds\n', elapsed_seconds);
    fprintf('[Grid Search] RMSECV: %.4f | WD: %.4f | RMSEP: %.4f | R2: %.4f | RPD: %.4f\n', ...
        summary.rmsecv, summary.wasserstein, target_rmsep, target_r2, target_rpd);
    fprintf('[Grid Search] %s\n', char(summary.combo_name));
end

function comparison_table = buildComparisonTable(ga_summary, grid_summary)
    method = [ga_summary.method; grid_summary.method];
    candidate_count = [ga_summary.candidate_count; grid_summary.candidate_count];
    selected_lv = [ga_summary.lv; grid_summary.lv];
    rmsecv = [ga_summary.rmsecv; grid_summary.rmsecv];
    wasserstein = [ga_summary.wasserstein; grid_summary.wasserstein];
    selection_score = [ga_summary.selection_score; grid_summary.selection_score];
    target_rmsep = [ga_summary.target_rmsep; grid_summary.target_rmsep];
    target_r2 = [ga_summary.target_r2; grid_summary.target_r2];
    target_rpd = [ga_summary.target_rpd; grid_summary.target_rpd];
    elapsed_seconds = [ga_summary.elapsed_seconds; grid_summary.elapsed_seconds];
    seconds_per_model = [ga_summary.seconds_per_model; grid_summary.seconds_per_model];
    combo = [ga_summary.combo_name; grid_summary.combo_name];

    comparison_table = table(method, candidate_count, selected_lv, rmsecv, ...
        wasserstein, selection_score, target_rmsep, ...
        target_r2, target_rpd, elapsed_seconds, seconds_per_model, combo);
end

function grid_selection_table = buildGridSelectionTable(grid_summary)
    candidate_index = grid_summary.candidate_index;
    lv = grid_summary.lv;
    rmsecv = grid_summary.rmsecv;
    wasserstein = grid_summary.wasserstein;
    proposed_score = grid_summary.selection_score;
    target_rmsep = grid_summary.target_rmsep;
    target_r2 = grid_summary.target_r2;
    target_rpd = grid_summary.target_rpd;
    total_models = grid_summary.candidate_count;
    elapsed_seconds = grid_summary.elapsed_seconds;
    seconds_per_model = grid_summary.seconds_per_model;
    combo = grid_summary.combo_name;

    grid_selection_table = table(candidate_index, lv, rmsecv, ...
        wasserstein, proposed_score, target_rmsep, target_r2, target_rpd, ...
        total_models, elapsed_seconds, seconds_per_model, combo);
end

function plotSearchComparison(comparison_table)
    method_labels = {'GA', 'Grid search'};
    bar_colors = [0.05, 0.20, 0.55; 0.70, 0.10, 0.10];

    runtime_fig = figure('Name', 'GA and Grid Search Runtime', ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [3, 3, 10.16, 10.16]);
    runtime_ax = axes(runtime_fig);
    runtime_values = comparison_table.elapsed_seconds;
    hold(runtime_ax, 'on');
    for i = 1:numel(runtime_values)
        bar(runtime_ax, i, runtime_values(i), 0.62, ...
            'FaceColor', bar_colors(i, :), ...
            'EdgeColor', 'none', ...
            'DisplayName', method_labels{i});
    end
    xticks(runtime_ax, 1:2);
    xticklabels(runtime_ax, method_labels);
    ylabel(runtime_ax, 'Elapsed time (s)', 'FontSize', 14, 'FontWeight', 'normal');
    legend(runtime_ax, method_labels, 'Location', 'northwest', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 12);
    addBarLabels(runtime_ax, runtime_values, '%.3f');
    styleComparisonAxis(runtime_ax);

    rmse_fig = figure('Name', 'GA and Grid Search RMSE Comparison', ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [14, 3, 10.16, 10.16]);
    rmse_ax = axes(rmse_fig);
    rmse_values = [comparison_table.rmsecv'; comparison_table.target_rmsep'];
    b_rmse = bar(rmse_ax, rmse_values, 0.72, 'grouped', 'EdgeColor', 'none');
    for i = 1:numel(b_rmse)
        b_rmse(i).FaceColor = bar_colors(i, :);
    end
    xticks(rmse_ax, 1:2);
    xticklabels(rmse_ax, {'RMSECV', 'RMSEP'});
    ylabel(rmse_ax, 'RMSE', 'FontSize', 14, 'FontWeight', 'normal');
    legend(rmse_ax, method_labels, 'Location', 'northwest', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 12);
    addGroupedBarLabels(rmse_ax, b_rmse, '%.3f');
    styleComparisonAxis(rmse_ax);
end

function addBarLabels(ax, values, format_str)
    y_max = max(values);
    if y_max <= 0
        y_max = 1;
    end
    ylim(ax, [0, y_max * 1.22]);

    for j = 1:numel(values)
        text(ax, j, values(j) + y_max * 0.035, sprintf(format_str, values(j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 11);
    end
end

function addGroupedBarLabels(ax, bar_handles, format_str)
    y_max = 0;
    for i = 1:numel(bar_handles)
        y_max = max(y_max, max(bar_handles(i).YData));
    end
    if y_max <= 0
        y_max = 1;
    end
    ylim(ax, [0, y_max * 1.24]);

    for i = 1:numel(bar_handles)
        x_values = bar_handles(i).XEndPoints;
        y_values = bar_handles(i).YEndPoints;
        labels = compose(format_str, y_values);
        text(ax, x_values, y_values + y_max * 0.035, labels, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 10);
    end
end

function styleComparisonAxis(ax)
    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridColor = [0.85, 0.85, 0.85];
    ax.GridAlpha = 0.55;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.LineWidth = 1.0;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 13;
    ax.XColor = [0.15, 0.15, 0.15];
    ax.YColor = [0.15, 0.15, 0.15];
end

function combo_matrix = buildComboMatrix()
    global Configuration;
    n_categories = numel(Configuration.Backbone);
    option_counts = zeros(1, n_categories);
    for k = 1:n_categories
        option_counts(k) = numel(Configuration.Backbone{k});
    end

    grids = cell(1, n_categories);
    grid_inputs = cell(1, n_categories);
    for k = 1:n_categories
        grid_inputs{k} = 1:option_counts(k);
    end

    [grids{:}] = ndgrid(grid_inputs{:});
    combo_matrix = zeros(numel(grids{1}), n_categories);
    for k = 1:n_categories
        combo_matrix(:, k) = grids{k}(:);
    end
end

function wasserstein = evaluateWasserstein(X_source, y_source, X_target, combo, lv)
    [Xs, Xt] = applyCombo(X_source, X_target, combo);
    [~,~,~,~,BETA] = plsregress(Xs, y_source, lv);
    yhat_source = [ones(size(Xs, 1), 1), Xs] * BETA;
    yhat_target = [ones(size(Xt, 1), 1), Xt] * BETA;
    wasserstein = calc_wasserstein(yhat_source, yhat_target);
end

function [rmsep, r2, rpd] = evaluateTargetPerformance(X_source, y_source, X_target_test, y_target_test, combo, lv)
    [Xs, Xt] = applyCombo(X_source, X_target_test, combo);
    [~,~,~,~,BETA] = plsregress(Xs, y_source, lv);
    yhat = [ones(size(Xt, 1), 1), Xt] * BETA;
    rmsep = sqrt(mean((yhat - y_target_test).^2));
    sst = sum((y_target_test - mean(y_target_test)).^2);
    sse = sum((yhat - y_target_test).^2);
    r2 = 1 - sse / sst;
    rpd = std(y_target_test) / rmsep;
end

function [Xs, Xt] = applyCombo(X_source, X_target, combo)
    global Configuration;
    Xs = X_source;
    Xt = X_target;
    for k = 1:numel(combo)
        [Xs, Xt] = Configuration.Backbone{k}{combo(k)}(Xs, Xt);
    end
end

function normalized = normalizeMinMax(values)
    min_value = min(values);
    max_value = max(values);
    if max_value == min_value
        normalized = zeros(size(values));
    else
        normalized = (values - min_value) / (max_value - min_value);
    end
end

function name = getComboName(combo)
    global Configuration;
    parts = strings(1, numel(combo));
    for k = 1:numel(combo)
        func_str = func2str(Configuration.Backbone{k}{combo(k)});
        func_str = regexprep(func_str, '^(smothing_|scatter_|baseline_|scaling_|centering_)', '');
        parts(k) = string(func_str);
    end
    name = strjoin(parts, " + ");
end
