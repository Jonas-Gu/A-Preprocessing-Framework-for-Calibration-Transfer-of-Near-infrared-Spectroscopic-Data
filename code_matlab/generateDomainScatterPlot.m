% generateDomainScatterPlot.m (域间任务专属脚本)
% ---------------------------------------------
% 绘制测试集预测值与实际值的散点图。
% 风格：SCIPlot (Times New Roman, Clean style)

% --- 0. 风格定义 (保持统一) ---
% 方案 A：海军蓝 (经典，极深)
%c_dot = [0.05, 0.20, 0.55]; 
% 方案 B：宝石蓝 (稍亮，但也很有质感)
%c_dot = [0.00, 0.30, 0.70];
% 方案 A：深紫罗兰 (推荐，优雅且清晰)
%c_dot = [0.45, 0.15, 0.60]; 
% 方案 B：暗紫色 (偏黑，更严肃)
%c_dot = [0.30, 0.05, 0.50];
% 方案 A：科研橙 (MATLAB/Python 默认风格，不刺眼)
c_dot = [0.85, 0.33, 0.10]; 
% 方案 B：亮橙色 (鲜艳，适合强调)
%c_dot = [0.95, 0.50, 0.10];

%c_dot = [0.1, 0.4, 0.8];   % 统一使用之前定义的深蓝色
c_line = [0.2, 0.2, 0.2];  % 参考线颜色 (深灰，比纯黑更柔和)

% 检查是否存在自定义标题
if ~exist('plotTitle', 'var') || isempty(plotTitle)
    plot_title_str = 'Domain Transfer Prediction';
else
    plot_title_str = plotTitle;
end

figure; % 创建一个新的图窗
set(gcf, 'Name', plot_title_str, 'NumberTitle', 'off', 'Color', 'w'); % 确保背景白

% --- 计算坐标轴范围 (逻辑保持不变) ---
min_global = min([ytest; yhat]);
max_global = max([ytest; yhat]);
range = max_global - min_global;
padding = range * 0.05; 
lim_min = min_global - padding;
lim_max = max_global + padding;
axis_limits = [lim_min, lim_max];

% --- 绘制参考线 (放在底层) ---
hold on;
plot(axis_limits, axis_limits, '--', 'Color', c_line, 'LineWidth', 1.5);

% --- 绘制散点图 (修改颜色和透明度) ---
scatter(ytest, yhat, 60, c_dot, 'filled', ...
    'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'none');

% --- 设置坐标轴和标签 ---
xlim(axis_limits);
ylim(axis_limits);

% 移除 Latex 解释器，以确保使用 SCIPlot 定义的 Times New Roman 字体
xlabel('Reference Value', 'FontWeight', 'bold'); 
ylabel('Predicted Value', 'FontWeight', 'bold');

% --- 应用动态标题 ---
% title(plot_title_str, 'Interpreter', 'none', 'FontSize', 16); % 论文图通常不需要标题，如需请取消注释

% --- 添加文本框 (逻辑保持不变) ---
text_str = sprintf('RMSEP = %.3f\nR^2 = %.3f\nRPD = %.3f', metric_p, R2_p, RPD_p);
text_x = lim_min + 0.05 * (lim_max - lim_min); % 稍微调整边距使其更美观
text_y = lim_max - 0.05 * (lim_max - lim_min); 

text(text_x, text_y, text_str, ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
     'FontSize', 14, ...              % 字号稍微加大以匹配 SCIPlot
     'FontName', 'Times New Roman', ...
     'BackgroundColor', 'w', ...
     'EdgeColor', 'k', ...
     'LineWidth', 1);                 % 文本框边框加粗

SCIPlot(); % 应用统一风格

% 强制坐标轴为正方形，这对于 "预测 vs 真实" 图非常重要
axis square; 

% --- 清除 plotTitle 变量 ---
clear plotTitle;
hold off;
