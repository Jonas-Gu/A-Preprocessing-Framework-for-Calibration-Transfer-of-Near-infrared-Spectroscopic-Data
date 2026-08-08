function [final_population, final_scores, x, fval] = gaDomainPreprocessing(X, y, LV)
    global Configuration;
    Configuration.XtrainData = X;
    Configuration.ytrainData = y;
    Configuration.Iter = 1; 
    Configuration.RMSECV =[];
    Configuration.minVal = 1000000;
    Configuration.minRealRMSECV = 1000000;
    Configuration.LVs = LV; % GA内部计算适应度时使用这个固定的LV
    
    nMaxComb = length(Configuration.Backbone);
    lb = ones(1, nMaxComb);
    ub = nMaxComb * ones(1, nMaxComb);
    for i = 1:nMaxComb
        ub(i) = length(Configuration.Backbone{i});
    end
    
    fitness_function = @(x) FitnessFunction(x);
    intcon = 1:nMaxComb;
    
    rng(0); % 保持随机种子
    
    options = optimoptions('ga');
    FunctionTolerance_Data = 1e-2;
    options = optimoptions(options,'PopulationSize', 16);
    
    options = optimoptions(options,'FunctionTolerance', FunctionTolerance_Data);
    options = optimoptions(options,'Display', 'iter'); % 显示GA进度
    options = optimoptions(options,'MaxTime',600);
    options = optimoptions(options,'MaxGenerations', 16);
    
    % 调用GA，并捕获最后两个输出：最终种群和分数
    [x, fval, exitflag, output, final_population, final_scores] = ga(fitness_function, nMaxComb, [], [], [], [], lb, ub, [], intcon, options);
end
