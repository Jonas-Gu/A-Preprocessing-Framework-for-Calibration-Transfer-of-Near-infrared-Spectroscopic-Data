function [best_combo_info, all_models_data] = preprocessCombos_domain(X_source, y_source)
    % PREPROCESSCOMBOS_DOMAIN
    % 
    % 逻辑：
    % 1. GA 搜索：仅优化 RMSECV (带惩罚)
    % 2. 迁移性评估：计算 Wasserstein 距离
    % 3. 最终择优：proposed 策略使用归一化 RMSECV 与 Wasserstein 距离的平方和
    
    global Configuration;
    
    % --- Selection strategy ---
    if isfield(Configuration, 'task')
        task = Configuration.task;
    else
        task = 'proposed';
    end
    
    fprintf('\n========================================\n');
    fprintf('[System] Strategy: %s\n', task);
    fprintf('========================================\n');
    
    LV_max = Configuration.LVs;
    original_LVs = Configuration.LVs;
    candidate_models = []; 
    
    % 搜索范围 (建议 2:LV_max 以包含简单模型)
    search_LVs = 6:LV_max; 
    
    % --- Stage 1: GA 单目标优化 (寻找高精度候选集) ---
    fprintf('Step 1: Running GA (Optimizing RMSECV)...\n');
    
    for i = 1:length(search_LVs)
        l = search_LVs(i);
        Configuration.LVs = l;
        
        % 运行标准 GA
        [population, scores, ~, ~] = gaDomainPreprocessing(X_source, y_source, l);
        
        % 简单去重
        [unique_pop, unique_idx] = unique(population, 'rows');
        unique_scores = scores(unique_idx);
        
        for k = 1:size(unique_pop, 1)
            combo_vec = unique_pop(k, :);
            rmsecv_val = unique_scores(k);
            
            % 还原真实的 RMSECV (去除惩罚因子)
            real_rmsecv = unpenalizedRMSECV(rmsecv_val, combo_vec);
            
            if ~isnan(real_rmsecv) && ~isinf(real_rmsecv)
                candidate_models(end+1).combo = combo_vec;
                candidate_models(end).lv = l;
                candidate_models(end).rmsecv = real_rmsecv;
                % Wasserstein 暂未计算
            end
        end
    end
    
    if isempty(candidate_models)
        error('GA 未找到任何有效模型。');
    end
    
    % --- Stage 2: 批量计算 Wasserstein 距离 ---
    fprintf('Step 2: Calculating Wasserstein Distance for %d candidates...\n', length(candidate_models));
    
    X_target_raw = Configuration.Xtarget; 
    
    for i = 1:length(candidate_models)
        f_combo = candidate_models(i).combo;
        l = candidate_models(i).lv;
        
        % 预处理 (全量数据，速度最快)
        Xs = X_source; 
        Xt = X_target_raw;
        for k = 1:length(f_combo)
            idx = f_combo(k);
            [Xs, Xt] = Configuration.Backbone{k}{idx}(Xs, Xt);
        end
        
        % 快速建模预测
        [~,~,~,~,BETA] = plsregress(Xs, y_source, l);
        yhat_s = [ones(size(Xs,1),1), Xs] * BETA;
        yhat_t = [ones(size(Xt,1),1), Xt] * BETA;
        
        % 计算距离
        candidate_models(i).wass_dist = calc_wasserstein(yhat_s, yhat_t);
        
        if mod(i, 50) == 0, fprintf('.'); end % 进度点
    end
    fprintf('\n');
    
    Configuration.LVs = original_LVs; 
    
    % --- Stage 3: Model selection strategy ---
    all_rmsecv = [candidate_models.rmsecv]';
    all_wass = [candidate_models.wass_dist]';
    
    if strcmpi(task, 'RMSECV')
        % 对照组：仅看 RMSECV
        [~, best_idx] = min(all_rmsecv);
    else
        % 1. 归一化 (Min-Max)
        min_cv = min(all_rmsecv); max_cv = max(all_rmsecv);
        min_wass = min(all_wass); max_wass = max(all_wass);
        
        % 防止分母为0
        if max_cv == min_cv, norm_rmsecv = zeros(size(all_rmsecv)); 
        else, norm_rmsecv = (all_rmsecv - min_cv) / (max_cv - min_cv); end
        
        if max_wass == min_wass, norm_wass = zeros(size(all_wass)); 
        else, norm_wass = (all_wass - min_wass) / (max_wass - min_wass); end
        
        if ~strcmpi(task, 'proposed')
            warning('Unknown task "%s"; using proposed strategy.', task);
        end

        % Proposed strategy: minimize the normalized squared distance to
        % the ideal point where both RMSECV and Wasserstein distance are 0.
        proposed_score = norm_rmsecv.^2 + norm_wass.^2;
        [~, best_idx] = min(proposed_score);
        fprintf('   [Logic] Selected model by proposed normalized squared-distance score.\n');
    end
    
    best_model = candidate_models(best_idx);
    
    % --- 输出结果 ---
    fprintf('\n--- Final Selection ---\n');
    fprintf('Index: %d\nRMSECV: %.4f\nWasserstein: %.4f\nLV: %d\n', ...
        best_idx, best_model.rmsecv, best_model.wass_dist, best_model.lv);
    
    best_combo_info.x1 = best_model.combo;
    best_combo_info.LV = best_model.lv;
    best_combo_info.rmsecv = best_model.rmsecv;
    best_combo_info.wass_dist = best_model.wass_dist;
    
    % 返回全量数据供画图使用
    all_models_data.rmsecv = all_rmsecv;
    all_models_data.wass_dist = all_wass;
    all_models_data.best_idx = best_idx;
    all_models_data.details = candidate_models;
    if exist('proposed_score', 'var')
        all_models_data.proposed_score = proposed_score;
    end
end
