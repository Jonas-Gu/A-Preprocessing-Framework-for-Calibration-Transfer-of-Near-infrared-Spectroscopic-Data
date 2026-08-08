function generateDomainPlots(plotData)
    % 该脚本绘制域间任务的关键图表 (Top-5 深蓝/深红对比版)
    % 风格同步：SCIPlot (Times New Roman, Box off, Clean style)
    
    % --- 0. 统一颜色定义 ---
    c_blue = [0.05, 0.20, 0.55]; % 深蓝 (代表 RMSECV 筛选组)
    c_red  = [0.70, 0.10, 0.10]; % 深红 (代表 proposed 筛选组)
    
    c_source_fill = [c_blue, 0.3]; 
    c_target_fill = [c_red, 0.5]; 
    
    % --- 1. 解包数据 ---
    qualified_models = plotData.qualified_models; 
    y_source = plotData.y_source;
    y_hat_source_best = plotData.y_hat_source_best;
    y_hat_target_best = plotData.y_hat_target_best;
    X_source_raw = plotData.X_source_raw;
    X_target_raw = plotData.X_target_raw;
    X_source_processed_best = plotData.X_source_processed_best;
    X_target_processed_best = plotData.X_target_processed_best;
    
    if isfield(plotData, 'top5_cv_idx') && isfield(plotData, 'top5_proposed_idx')
        top5_cv_idx = plotData.top5_cv_idx;
        top5_proposed_idx = plotData.top5_proposed_idx;
        has_top5_data = true;
    else
        has_top5_data = false;
    end

    % --- 图 1：源域真实Y分布 ---
    figure; set(gcf, 'Name', 'Source Y Distribution', 'Color', 'w');
    histogram(y_source, 'Normalization', 'probability', 'FaceColor', c_blue, 'FaceAlpha', 0.7, 'EdgeColor', 'none');
    xlabel('Reference Value (Y)'); ylabel('Probability'); SCIPlot();

    % --- 图 2：预测值分布 (直方图 + 曲线) ---
    figure; set(gcf, 'Name', 'Predicted Y Distribution', 'Color', 'w'); hold on;
    [f_s, xi_s] = ksdensity(y_hat_source_best);
    [f_t, xi_t] = ksdensity(y_hat_target_best);
    % 直方图
    histogram(y_hat_source_best, 'Normalization', 'pdf', 'FaceColor', c_blue, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    histogram(y_hat_target_best, 'Normalization', 'pdf', 'FaceColor', c_red, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % 曲线
    h_line_S = plot(xi_s, f_s, 'Color', c_blue, 'LineWidth', 2.5, 'DisplayName', 'Source');
    h_line_T = plot(xi_t, f_t, 'Color', c_red, 'LineWidth', 2.5, 'DisplayName', 'Target');
    xlabel('Predicted Value'); ylabel('Probability Density');
    legend([h_line_S, h_line_T], 'Location', 'northeast', 'Box', 'off'); SCIPlot(); hold off;

    % --- 图 3：原始光谱 ---
    figure; set(gcf, 'Name', 'Raw Spectra', 'Color', 'w'); hold on;
    h1 = plot(X_source_raw', 'Color', c_source_fill, 'LineWidth', 0.5);
    h2 = plot(X_target_raw', 'Color', c_target_fill, 'LineWidth', 0.5);
    for k=1:numel(h1), h1(k).HandleVisibility='off'; end
    for k=1:numel(h2), h2(k).HandleVisibility='off'; end
    legend([plot(nan,nan,'Color',c_blue,'LineWidth',2), plot(nan,nan,'Color',c_red,'LineWidth',2)], ...
           {'Source Domain', 'Target Domain'}, 'Location','northeast','Box','off');
    xlabel('Variable'); ylabel('Absorbance'); axis tight; SCIPlot(); hold off;

    % --- 图 4：预处理光谱 ---
    figure; set(gcf, 'Name', 'Processed Spectra', 'Color', 'w'); hold on;
    h1 = plot(X_source_processed_best', 'Color', c_source_fill, 'LineWidth', 0.5);
    h2 = plot(X_target_processed_best', 'Color', c_target_fill, 'LineWidth', 0.5);
    for k=1:numel(h1), h1(k).HandleVisibility='off'; end
    for k=1:numel(h2), h2(k).HandleVisibility='off'; end
    legend([plot(nan,nan,'Color',c_blue,'LineWidth',2), plot(nan,nan,'Color',c_red,'LineWidth',2)], ...
           {'Source', 'Target'}, 'Location','northeast','Box','off');
    xlabel('Variable'); ylabel('Absorbance'); axis tight; SCIPlot(); hold off;

    % =========================================================================
    % Top-5 对比柱状图 (颜色同步: 深蓝=RMSECV组, 深红=proposed组)
    % =========================================================================
    if has_top5_data
        % 提取数据
        cv_rmsecv = [qualified_models(top5_cv_idx).rmsecv];
        cv_rmsep  = [qualified_models(top5_cv_idx).rmsep];
        
        proposed_rmsecv = [qualified_models(top5_proposed_idx).rmsecv];
        proposed_rmsep  = [qualified_models(top5_proposed_idx).rmsep];
        
        n_bars = length(cv_rmsecv);
        
        % --- 图 7: 源域性能对比 (RMSECV) ---
        figure; set(gcf, 'Name', 'Top-5 Comparison (Source RMSECV)', 'Color', 'w'); hold on;
        b1 = bar(1:n_bars, [cv_rmsecv(:), proposed_rmsecv(:)], 'grouped');
        % 设置颜色
        b1(1).FaceColor = c_blue; b1(1).EdgeColor = 'none'; % 组1: RMSECV选出的
        b1(2).FaceColor = c_red;  b1(2).EdgeColor = 'none'; % 组2: proposed选出的
        
        ylabel('RMSECV (Source)'); xlabel('Rank');
        legend({'Selected by RMSECV', 'Selected by proposed'}, 'Location', 'northwest', 'Box', 'off');
        set(gca, 'XTick', 1:n_bars); grid on; SCIPlot(); hold off;
        
        % --- 图 8: 目标域性能对比 (RMSEP) ---
        figure; set(gcf, 'Name', 'Top-5 Comparison (Target RMSEP)', 'Color', 'w'); hold on;
        b2 = bar(1:n_bars, [cv_rmsep(:), proposed_rmsep(:)], 'grouped');
        % 设置颜色
        b2(1).FaceColor = c_blue; b2(1).EdgeColor = 'none';
        b2(2).FaceColor = c_red;  b2(2).EdgeColor = 'none';
        
        ylabel('RMSEP (Target)'); xlabel('Rank');
        legend({'Selected by RMSECV', 'Selected by proposed'}, 'Location', 'northwest', 'Box', 'off');
        set(gca, 'XTick', 1:n_bars); grid on; SCIPlot(); hold off;
    end
end

% --- 辅助函数 ---
function SCIPlot()
    grid off; box on; a = gca; a.FontSize = 14; a.FontName = 'Times New Roman'; a.LineWidth = 1.2;
    set(gca, 'LooseInset', [0.01 0.01 0.01 0.01]);
end
