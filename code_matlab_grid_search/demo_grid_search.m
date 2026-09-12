clear;
close all;
global Configuration;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', 'code_matlab'));

% ---Configuration---
%sourceDataset = "wheatA1_20048"; 
%targetDataset = "wheatA3_48200"; 
sourceDataset = "nirshootout1"; 
targetDataset = "nirshootout2"; 
task = 'proposed';
fold = 4;      
LV = 10; 

% 'proposed' : select by normalized RMSECV and Wasserstein distance.
% 'RMSECV'   : select only by source-domain cross-validation accuracy.
% This grid-search demo keeps the proposed RMSECV-WD selection rule but
% replaces the GA candidate-pool generation step with exhaustive enumeration.



tStart = tic;
Configuration = []; 

% --- 1. 环境初始化 ---
fprintf('Model selection strategy: %s\n', task);
fprintf('源域数据集: %s\n', sourceDataset);
fprintf('目标域数据集: %s\n', targetDataset);

load(sourceDataset);
if exist('Xtest', 'var') && exist('ytest', 'var')
    source_test_exists = true;
else
    source_test_exists = false;
end

target_data = load(targetDataset);
buildConfigration(task, fold, LV);

% 配置全局变量
Configuration.Xtarget = target_data.X;     % (目标域训练集, 用于 Wasserstein 计算)
Configuration.Xtarget_RAW = target_data.X; 
Configuration.XtrainData = X;              
Configuration.ytrainData = y;              
Configuration.Xtarget_test = target_data.Xtest;
Configuration.ytarget_test = target_data.ytest;

Xtesttarget = target_data.Xtest;
ytesttarget = target_data.ytest;
clear target_data;
    
% --- 2. 核心调用 (获取最优模型 & 所有候选模型数据) ---
% 接收两个返回值：best_info 和 all_results
[best_model_info, optimization_results] = preprocessCombos_domain_grid_search(X, y);

% --- 3. [新增] 论文关键图表：帕累托前沿可视化 ---
figure('Name', 'Proposed Selection Map', 'Color', 'w');
sc_handle = scatter(optimization_results.wass_dist, optimization_results.rmsecv, ...
    60, 'filled', 'MarkerFaceColor', [0.05, 0.20, 0.55], 'MarkerEdgeColor', 'none'); 
hold on;

% 标记选中的最优模型
best_idx = optimization_results.best_idx;
scatter(optimization_results.wass_dist(best_idx), optimization_results.rmsecv(best_idx), ...
    120, 'p', 'filled', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');

% 美化图表
grid on; box on;
xlabel('Distribution Discrepancy (Wasserstein Distance)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Source Domain Accuracy (RMSECV)', 'FontSize', 12, 'FontWeight', 'bold');
legend({'Candidate Models', 'Selected Optimal Model'}, 'Location', 'northeast');

% --- 4. 验证最优模型在目标域的性能 ---
X_source_processed = X;
X_target_test_processed = Xtesttarget;

% 复现预处理步骤
for k = 1:length(best_model_info.x1)
    idx = best_model_info.x1(k);
    preproc_func = Configuration.Backbone{k}{idx};
    [X_source_processed, X_target_test_processed] = preproc_func(X_source_processed, X_target_test_processed);
end

% 训练最终模型
[~,~,~,~,BETA] = plsregress(X_source_processed, y, best_model_info.LV);

% 预测
yhat_target = [ones(size(X_target_test_processed,1),1), X_target_test_processed] * BETA;

% 计算指标
RMSEP_transfer = sqrt(mean((yhat_target - ytesttarget).^2));
SST = sum((ytesttarget - mean(ytesttarget)).^2);
SSE = sum((yhat_target - ytesttarget).^2);
R2_transfer = 1 - (SSE / SST);
RPD_transfer = std(ytesttarget) / RMSEP_transfer;

fprintf("\n--- 模型转移验证结果 (目标域测试集) ---\n");
fprintf("RMSEP: %.4f\n", RMSEP_transfer);
fprintf("R2:    %.4f\n", R2_transfer);
fprintf("RPD:   %.4f\n", RPD_transfer);

% 绘制预测值散点图
ytest = ytesttarget;
yhat = yhat_target;
metric_p = RMSEP_transfer;
R2_p = R2_transfer;
RPD_p = RPD_transfer;
plotTitle = 'Prediction on Target Domain Test Set';
generateDomainScatterPlot; 

% --- 5. 源域测试集回测 (保持不变) ---
if source_test_exists
    fprintf('\n--- 源域测试集回测 ---\n');
    src_raw = load(sourceDataset);
    X_src_train_temp = src_raw.X;
    X_src_test_temp  = src_raw.Xtest;
    y_src_test_temp  = src_raw.ytest;
    
    for k = 1:length(best_model_info.x1)
        idx = best_model_info.x1(k);
        preproc_func = Configuration.Backbone{k}{idx};
        [X_src_train_temp, X_src_test_temp] = preproc_func(X_src_train_temp, X_src_test_temp);
    end
    
    yhat_source_test = [ones(size(X_src_test_temp,1),1), X_src_test_temp] * BETA;
    
    RMSEP_source = sqrt(mean((yhat_source_test - y_src_test_temp).^2));
    SST_source = sum((y_src_test_temp - mean(y_src_test_temp)).^2);
    SSE_source = sum((yhat_source_test - y_src_test_temp).^2);
    R2_source = 1 - (SSE_source / SST_source);
    RPD_source = std(y_src_test_temp) / RMSEP_source;
    fprintf("Source RMSEP: %.4f\n", RMSEP_source);
    fprintf("Source R2:    %.4f\n", R2_source);
    fprintf("Source RPD:   %.4f\n", RPD_source);
    
    ytest = y_src_test_temp;
    yhat = yhat_source_test;
    metric_p = RMSEP_source;
    R2_p = R2_source;
    RPD_p = RPD_source;
    plotTitle = 'Prediction on Source Domain Test Set';
    generateDomainScatterPlot;
end

% =========================================================================
% 6. 生成高级分析图表 & Top-5 性能对比
% =========================================================================
fprintf('\n========================================\n');
fprintf('[Advanced Analysis] Generating Top-5 Comparisons...\n');

% 1. 准备基础绘图数据
plotData = struct();
details = optimization_results.details; 
plotData.y_source = y;
plotData.X_source_raw = X;
plotData.X_target_raw = Configuration.Xtarget_RAW;

% 2. 筛选 Top-5 索引 (核心修正部分)

% -------------------------------------------------------------------------
% Group A: 传统方法 (Benchmark / RMSECV)
% 逻辑：基于【全部候选模型】，仅根据 RMSECV 从小到大排序
% -------------------------------------------------------------------------
all_rmsecv_global = [details.rmsecv]; % 获取所有模型的 RMSECV
[~, idx_cv_sorted] = sort(all_rmsecv_global, 'ascend'); % 全局排序
top5_cv_idx = idx_cv_sorted(1:min(5, length(details))); % 取前5

% -------------------------------------------------------------------------
% Group B: 本文方法 (Proposed / Sum of Squares Strategy)
% 逻辑：
% 1. 对 RMSECV 和 Wasserstein 距离进行归一化
% 2. 计算两者的平方和
% 3. 直接根据平方和从小到大排序，取前 5 名
% -------------------------------------------------------------------------
all_rmsecv = [details.rmsecv]';
all_wass   = [details.wass_dist]';

% 归一化 (Min-Max)
min_cv = min(all_rmsecv); max_cv = max(all_rmsecv);
min_wass = min(all_wass); max_wass = max(all_wass);

if max_cv == min_cv, norm_cv = zeros(size(all_rmsecv)); 
else, norm_cv = (all_rmsecv - min_cv) / (max_cv - min_cv); end

if max_wass == min_wass, norm_wass = zeros(size(all_wass)); 
else, norm_wass = (all_wass - min_wass) / (max_wass - min_wass); end

% 直接基于平方和进行排序
sum_of_squares = norm_cv.^2 + norm_wass.^2;
[~, sorted_indices] = sort(sum_of_squares, 'ascend');

% 获取 Top-5 索引
top5_proposed_idx = sorted_indices(1:min(5, length(details)));

% 3. 在命令行打印组合名称 (Paper Table Material)
fprintf('\n--- [Group A] Top 5 Models by RMSECV (Benchmark: Global Sort) ---\n');
for i = 1:length(top5_cv_idx)
    idx = top5_cv_idx(i);
    combo = details(idx).combo;
    str_name = getComboName(combo);
    fprintf('Rank %d: RMSECV=%.4f | Wass=%.4f | LV=%d | Combo: %s\n', ...
        i, details(idx).rmsecv, details(idx).wass_dist, details(idx).lv, str_name);
end

fprintf('\n--- [Group B] Top 5 Models by Proposed Strategy ---\n');
for i = 1:length(top5_proposed_idx)
    idx = top5_proposed_idx(i);
    combo = details(idx).combo;
    str_name = getComboName(combo);
    fprintf('Rank %d: RMSECV=%.4f | Wass=%.4f | LV=%d | Combo: %s\n', ...
        i, details(idx).rmsecv, details(idx).wass_dist, details(idx).lv, str_name);
end
fprintf('========================================\n');

% 4. 计算目标域 RMSEP (Oracle Validation)
calc_indices = unique([top5_cv_idx(:); top5_proposed_idx(:)]);
X_target_test = Configuration.Xtarget_test; 
y_target_test = Configuration.ytarget_test; 

for i = 1:length(calc_indices)
    idx = calc_indices(i);
    combo = details(idx).combo;
    lv = details(idx).lv;
    
    % 预处理
    Xs_temp = X; 
    Xt_test_temp = X_target_test;
    for k = 1:length(combo)
        [Xs_temp, Xt_test_temp] = Configuration.Backbone{k}{combo(k)}(Xs_temp, Xt_test_temp);
    end
    
    % 训练预测
    [~,~,~,~,B] = plsregress(Xs_temp, y, lv);
    yhat = [ones(size(Xt_test_temp,1),1), Xt_test_temp] * B;
    
    % 存储 RMSEP
    details(idx).rmsep = sqrt(mean((yhat - y_target_test).^2));
end

% 更新 plotData
plotData.qualified_models = details;
plotData.top5_cv_idx = top5_cv_idx;
plotData.top5_proposed_idx = top5_proposed_idx;

% 5. 准备最优模型的光谱和投影数据
X_s_best_plot = X;
X_t_best_plot = Configuration.Xtarget_RAW;
for k = 1:length(best_model_info.x1)
    idx = best_model_info.x1(k);
    [X_s_best_plot, X_t_best_plot] = Configuration.Backbone{k}{idx}(X_s_best_plot, X_t_best_plot);
end
plotData.X_source_processed_best = X_s_best_plot;
plotData.X_target_processed_best = X_t_best_plot;

[~,~,~,~,BETA_best] = plsregress(X_s_best_plot, y, best_model_info.LV);
plotData.y_hat_source_best = [ones(size(X_s_best_plot,1),1), X_s_best_plot] * BETA_best;
plotData.y_hat_target_best = [ones(size(X_t_best_plot,1),1), X_t_best_plot] * BETA_best;

% 6. 调用绘图
generateDomainPlots(plotData);

fprintf('\n[System] All plots generated.\n');
fprintf("Total elapsed time: %.2f seconds\n", toc(tStart));

% --- 辅助函数：获取组合名称 (显式显示 none) ---
function name = getComboName(combo)
    global Configuration;
    name = '';
    for k = 1:length(combo)
        func_handle = Configuration.Backbone{k}{combo(k)};
        func_str = func2str(func_handle);
        % 去掉 unnecessary 前缀
        func_str = regexprep(func_str, '^(smothing_|scatter_|baseline_|scaling_|centering_)', '');
        if k == 1
            name = func_str;
        else
            name = [name, ' + ', func_str];
        end
    end
end
