clear;
close all;
global Configuration;

sourceDataset = "nirshootout1";
targetDataset = "nirshootout2";
task = 'proposed';
fold = 4;
LV = 10;

Configuration = [];
load(sourceDataset);
target_data = load(targetDataset);
buildConfigration(task, fold, LV);

Configuration.Xtarget = target_data.X;
Configuration.Xtarget_RAW = target_data.X;
Configuration.XtrainData = X;
Configuration.ytrainData = y;
Configuration.Xtarget_test = target_data.Xtest;
Configuration.ytarget_test = target_data.ytest;

Xtarget_train = target_data.X;
Xtarget_test = target_data.Xtest;
ytarget_test = target_data.ytest;
clear target_data;

fprintf('\n[WD-N Sensitivity] Building candidate pool once...\n');
[~, optimization_results] = preprocessCombos_domain(X, y);
details = optimization_results.details;

n_values = unique([60, 80, 100, 120, 140, 160, 180, 200]);
rows = struct([]);

for i = 1:length(n_values)
    n_points = n_values(i);
    wass_values = zeros(length(details), 1);

    for j = 1:length(details)
        combo = details(j).combo;
        lv = details(j).lv;
        [Xs_train, Xt_train] = applyCombo(X, Xtarget_train, combo);
        [~,~,~,~,BETA] = plsregress(Xs_train, y, lv);
        yhat_source = [ones(size(Xs_train, 1), 1), Xs_train] * BETA;
        yhat_target = [ones(size(Xt_train, 1), 1), Xt_train] * BETA;
        wass_values(j) = calc_wasserstein(yhat_source, yhat_target, n_points);
    end

    rmsecv_values = [details.rmsecv]';
    score = normalizeMinMax(rmsecv_values).^2 + normalizeMinMax(wass_values).^2;
    [~, best_idx] = min(score);
    best = details(best_idx);

    [Xs_final, Xt_test_final] = applyCombo(X, Xtarget_test, best.combo);
    [~,~,~,~,BETA] = plsregress(Xs_final, y, best.lv);
    yhat_test = [ones(size(Xt_test_final, 1), 1), Xt_test_final] * BETA;

    rmsep = sqrt(mean((yhat_test - ytarget_test).^2));
    sst = sum((ytarget_test - mean(ytarget_test)).^2);
    sse = sum((yhat_test - ytarget_test).^2);
    r2 = 1 - sse / sst;
    rpd = std(ytarget_test) / rmsep;

    rows(i).n_points = n_points;
    rows(i).best_idx = best_idx;
    rows(i).lv = best.lv;
    rows(i).rmsecv = best.rmsecv;
    rows(i).wasserstein = wass_values(best_idx);
    rows(i).rmsep = rmsep;
    rows(i).r2 = r2;
    rows(i).rpd = rpd;
    rows(i).combo = string(getComboName(best.combo));

    fprintf('N=%d | idx=%d | LV=%d | RMSEP=%.4f | R2=%.4f | RPD=%.4f | %s\n', ...
        n_points, best_idx, best.lv, rmsep, r2, rpd, rows(i).combo);
end

result_table = struct2table(rows);
plotRmsecvRmsepVsN(result_table);
disp(result_table);

function [Xs, Xt] = applyCombo(X_source, X_target, combo)
    global Configuration;
    Xs = X_source;
    Xt = X_target;
    for k = 1:length(combo)
        [Xs, Xt] = Configuration.Backbone{k}{combo(k)}(Xs, Xt);
    end
end

function y = normalizeMinMax(x)
    min_x = min(x);
    max_x = max(x);
    if max_x == min_x
        y = zeros(size(x));
    else
        y = (x - min_x) / (max_x - min_x);
    end
end

function name = getComboName(combo)
    global Configuration;
    parts = strings(1, length(combo));
    for k = 1:length(combo)
        func_str = func2str(Configuration.Backbone{k}{combo(k)});
        func_str = regexprep(func_str, '^(smothing_|scatter_|baseline_|scaling_|centering_)', '');
        parts(k) = string(func_str);
    end
    name = strjoin(parts, " + ");
end

function plotRmsecvRmsepVsN(result_table)
    x = result_table.n_points(:);
    rmsecv = result_table.rmsecv(:);
    rmsep = result_table.rmsep(:);

    fig = figure('Name', 'WD N Sensitivity: RMSECV and RMSEP', ...
        'Color', 'w', 'Units', 'centimeters', 'Position', [3, 3, 14, 8.5]);
    ax = axes(fig);
    hold(ax, 'on');

    rmsecv_color = [0.00, 0.27, 0.55];
    rmsep_color = [0.74, 0.16, 0.14];

    plot(ax, x, rmsecv, '-o', ...
        'LineWidth', 1.7, 'MarkerSize', 6, ...
        'Color', rmsecv_color, ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', rmsecv_color, ...
        'DisplayName', 'RMSECV');
    plot(ax, x, rmsep, '-s', ...
        'LineWidth', 1.7, 'MarkerSize', 6, ...
        'Color', rmsep_color, ...
        'MarkerFaceColor', 'w', ...
        'MarkerEdgeColor', rmsep_color, ...
        'DisplayName', 'RMSEP');

    xlabel(ax, 'Number of quantile grid points (N)', ...
        'FontSize', 14, 'FontWeight', 'normal');
    ylabel(ax, 'RMSE', ...
        'FontSize', 14, 'FontWeight', 'normal');
    xticks(ax, x);

    y_all = [rmsecv; rmsep];
    y_min = min(y_all);
    y_max = max(y_all);
    y_range = y_max - y_min;
    bottom_pad = max(y_range * 0.20, max(abs(y_all)) * 0.01);
    top_pad = max(y_range * 0.45, max(abs(y_all)) * 0.04);
    if bottom_pad == 0
        bottom_pad = max(abs(y_min) * 0.05, 0.01);
    end
    if top_pad == 0
        top_pad = max(abs(y_max) * 0.10, 0.02);
    end
    ylim(ax, [y_min - bottom_pad, y_max + top_pad]);

    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridColor = [0.85, 0.85, 0.85];
    ax.GridAlpha = 0.55;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.LineWidth = 1.0;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 14;
    ax.XColor = [0.15, 0.15, 0.15];
    ax.YColor = [0.15, 0.15, 0.15];

    legend(ax, 'Location', 'northeast', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 12);
end
