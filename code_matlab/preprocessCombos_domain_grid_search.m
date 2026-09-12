function [best_combo_info, all_models_data] = preprocessCombos_domain_grid_search(X_source, y_source)
    % PREPROCESSCOMBOS_DOMAIN_GRID_SEARCH
    %
    % Grid-search variant of preprocessCombos_domain.m.
    % The downstream WD calculation and normalized RMSECV-WD model-selection
    % rule are kept unchanged; only the GA candidate-pool generation step is
    % replaced by exhaustive enumeration of the preprocessing-LV search space.

    global Configuration;

    if isfield(Configuration, 'task')
        task = Configuration.task;
    else
        task = 'proposed';
    end

    fprintf('\n========================================\n');
    fprintf('[System] Strategy: %s | candidate generation: grid search\n', task);
    fprintf('========================================\n');

    LV_max = Configuration.LVs;
    original_LVs = Configuration.LVs;
    search_LVs = 6:LV_max;
    combo_matrix = buildComboMatrix();
    total_models = size(combo_matrix, 1) * numel(search_LVs);

    candidate_models = [];
    candidate_models(total_models).combo = [];
    candidate_models(total_models).lv = [];
    candidate_models(total_models).rmsecv = [];
    candidate_models(total_models).wass_dist = [];

    fprintf('Step 1: Running exhaustive grid search over preprocessing-LV candidates...\n');
    fprintf('Preprocessing combinations: %d\n', size(combo_matrix, 1));
    fprintf('LV values: %s\n', strjoin(string(search_LVs), ', '));
    fprintf('Total candidates: %d\n', total_models);

    row_idx = 0;
    for i = 1:numel(search_LVs)
        l = search_LVs(i);
        Configuration.LVs = l;

        for j = 1:size(combo_matrix, 1)
            row_idx = row_idx + 1;
            combo_vec = combo_matrix(j, :);

            rmsecv_val = crossvalidate(X_source, y_source, combo_vec, ...
                1:size(X_source, 2), 0);

            candidate_models(row_idx).combo = combo_vec;
            candidate_models(row_idx).lv = l;
            candidate_models(row_idx).rmsecv = rmsecv_val;

            if mod(row_idx, 50) == 0
                fprintf('.');
            end
        end
        fprintf('\n');
    end

    valid_idx = arrayfun(@(m) ~isnan(m.rmsecv) && ~isinf(m.rmsecv), candidate_models);
    candidate_models = candidate_models(valid_idx);

    if isempty(candidate_models)
        error('Grid search did not find any valid model.');
    end

    fprintf('Step 2: Calculating Wasserstein Distance for %d candidates...\n', length(candidate_models));

    X_target_raw = Configuration.Xtarget;

    for i = 1:length(candidate_models)
        f_combo = candidate_models(i).combo;
        l = candidate_models(i).lv;

        Xs = X_source;
        Xt = X_target_raw;
        for k = 1:length(f_combo)
            idx = f_combo(k);
            [Xs, Xt] = Configuration.Backbone{k}{idx}(Xs, Xt);
        end

        [~,~,~,~,BETA] = plsregress(Xs, y_source, l);
        yhat_s = [ones(size(Xs,1),1), Xs] * BETA;
        yhat_t = [ones(size(Xt,1),1), Xt] * BETA;

        candidate_models(i).wass_dist = calc_wasserstein(yhat_s, yhat_t);

        if mod(i, 50) == 0
            fprintf('.');
        end
    end
    fprintf('\n');

    Configuration.LVs = original_LVs;

    all_rmsecv = [candidate_models.rmsecv]';
    all_wass = [candidate_models.wass_dist]';

    if strcmpi(task, 'RMSECV')
        [~, best_idx] = min(all_rmsecv);
    else
        min_cv = min(all_rmsecv);
        max_cv = max(all_rmsecv);
        min_wass = min(all_wass);
        max_wass = max(all_wass);

        if max_cv == min_cv
            norm_rmsecv = zeros(size(all_rmsecv));
        else
            norm_rmsecv = (all_rmsecv - min_cv) / (max_cv - min_cv);
        end

        if max_wass == min_wass
            norm_wass = zeros(size(all_wass));
        else
            norm_wass = (all_wass - min_wass) / (max_wass - min_wass);
        end

        if ~strcmpi(task, 'proposed')
            warning('Unknown task "%s"; using proposed strategy.', task);
        end

        proposed_score = norm_rmsecv.^2 + norm_wass.^2;
        [~, best_idx] = min(proposed_score);
        fprintf('   [Logic] Selected model by proposed normalized squared-distance score.\n');
    end

    best_model = candidate_models(best_idx);

    fprintf('\n--- Final Selection ---\n');
    fprintf('Index: %d\nRMSECV: %.4f\nWasserstein: %.4f\nLV: %d\n', ...
        best_idx, best_model.rmsecv, best_model.wass_dist, best_model.lv);

    best_combo_info.x1 = best_model.combo;
    best_combo_info.LV = best_model.lv;
    best_combo_info.rmsecv = best_model.rmsecv;
    best_combo_info.wass_dist = best_model.wass_dist;

    all_models_data.rmsecv = all_rmsecv;
    all_models_data.wass_dist = all_wass;
    all_models_data.best_idx = best_idx;
    all_models_data.details = candidate_models;
    if exist('proposed_score', 'var')
        all_models_data.proposed_score = proposed_score;
    end
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
