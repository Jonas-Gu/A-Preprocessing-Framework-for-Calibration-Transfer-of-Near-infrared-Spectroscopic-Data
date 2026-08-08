function buildConfigration(task, fold, LV)
global Configuration;
Configuration.fold = fold;
Configuration.task = task; % model selection strategy: 'proposed' or 'RMSECV'
Configuration.LVs = LV;
% 判定是否配置了预处理方法顺序
if ~isfield(Configuration, 'Backbone') || isempty(Configuration.Backbone)
    Configuration.Backbone = {smothingFuncs(),baselineFuncs(),scatterFuncs(),centeringFuncs(),scalingFuncs()};
    %Configuration.Backbone = {scatterFuncs(),baselineFuncs(),smothingFuncs(),centeringFuncs(),scalingFuncs()};
    %Configuration.Backbone = {baselineFuncs(),scatterFuncs(),smothingFuncs(),centeringFuncs(),scalingFuncs()};
    disp('应用默认预处理顺序');
else
    disp('应用用户配置的方法');
    %disp(Configuration.Backbone);
end
Configuration.modeling_evalation = @pls_modeling_evalation;

