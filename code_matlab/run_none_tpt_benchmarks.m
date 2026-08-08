% Basic benchmark methods requested by the reviewer:
%   NONE: source-domain PLS without preprocessing or transfer.
%   TPT : target-predicting-target PLS with target labels.
%
% This script is independent from demo.m.

clear; clc; close all;
%sourceDataset = "wheatA1_20048"; 
%targetDataset = "wheatA3_48200"; 
sourceDataset = "nirshootout1";
targetDataset = "nirshootout2";
lv_range = 6:10;
source_cv_fold = 4;
tpt_cv_fold = 4;

source_data = load(sourceDataset);
target_data = load(targetDataset);

[none_lv, none_rmsecv] = selectLvByCv(source_data.X, source_data.y, lv_range, source_cv_fold);
[none_target_rmsep, none_target_r2, none_target_rpd] = evaluatePls( ...
    source_data.X, source_data.y, target_data.Xtest, target_data.ytest, none_lv);

[none_source_rmsep, none_source_r2, none_source_rpd] = evaluatePls( ...
    source_data.X, source_data.y, source_data.Xtest, source_data.ytest, none_lv);

[tpt_lv, tpt_rmsecv] = selectLvByCv(target_data.X, target_data.y, lv_range, tpt_cv_fold);
[tpt_target_rmsep, tpt_target_r2, tpt_target_rpd] = evaluatePls( ...
    target_data.X, target_data.y, target_data.Xtest, target_data.ytest, tpt_lv);

method = ["NONE"; "TPT"];
training_domain = ["source"; "target"];
cv_fold = [source_cv_fold; tpt_cv_fold];
selected_lv = [none_lv; tpt_lv];
rmsecv = [none_rmsecv; tpt_rmsecv];
target_rmsep = [none_target_rmsep; tpt_target_rmsep];
target_r2 = [none_target_r2; tpt_target_r2];
target_rpd = [none_target_rpd; tpt_target_rpd];
source_rmsep = [none_source_rmsep; NaN];
source_r2 = [none_source_r2; NaN];
source_rpd = [none_source_rpd; NaN];

benchmark_table = table(method, training_domain, cv_fold, selected_lv, rmsecv, ...
    target_rmsep, target_r2, target_rpd, source_rmsep, source_r2, source_rpd);

fprintf('\n[NONE/TPT Benchmarks]\n');
disp(benchmark_table);

function [best_lv, best_rmsecv] = selectLvByCv(X, y, lv_range, fold)
    rng(0);
    [XTrainData, YTrainData, XTestData, YTestData] = easyCrossvalidation(X, y, fold);
    rmsecv_by_lv = zeros(numel(lv_range), 1);

    for i = 1:numel(lv_range)
        lv = lv_range(i);
        fold_rmsep = zeros(fold, 1);

        for j = 1:fold
            X_train = XTrainData{j};
            y_train = YTrainData{j};
            X_val = XTestData{j};
            y_val = YTestData{j};

            [~,~,~,~,BETA] = plsregress(X_train, y_train, lv);
            yhat = [ones(size(X_val, 1), 1), X_val] * BETA;
            fold_rmsep(j) = sqrt(mean((yhat - y_val).^2));
        end

        rmsecv_by_lv(i) = mean(fold_rmsep);
    end

    [best_rmsecv, best_idx] = min(rmsecv_by_lv);
    best_lv = lv_range(best_idx);
end

function [rmsep, r2, rpd] = evaluatePls(X_train, y_train, X_test, y_test, lv)
    [~,~,~,~,BETA] = plsregress(X_train, y_train, lv);
    yhat = [ones(size(X_test, 1), 1), X_test] * BETA;
    rmsep = sqrt(mean((yhat - y_test).^2));
    sst = sum((y_test - mean(y_test)).^2);
    sse = sum((yhat - y_test).^2);
    r2 = 1 - sse / sst;
    rpd = std(y_test) / rmsep;
end
