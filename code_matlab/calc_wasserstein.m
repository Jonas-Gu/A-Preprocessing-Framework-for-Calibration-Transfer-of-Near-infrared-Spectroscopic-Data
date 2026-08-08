function dist = calc_wasserstein(Ys, Yt, n_points)
    % CALC_WASSERSTEIN 计算源域和目标域预测值之间的 1-Wasserstein 距离
    % 输入:
    %   Ys: 源域样本的预测值 (列向量)
    %   Yt: 目标域样本的预测值 (列向量)
    
    % 1. 确保输入为列向量
    Ys = Ys(:); 
    Yt = Yt(:);
    
    % 2. 统一采样点 (处理源域和目标域样本数不一致的情况)
    % 默认使用 100 个等间距分位点来描述分布形状，这等价于对 CDF 逆函数采样
    if nargin < 3 || isempty(n_points)
        n_points = 100;
    end
    prob_axis = linspace(0, 1, n_points)';
    
    % 3. 计算分位数 (Quantiles)
    % 这一步非常关键，它将两个不同样本量的分布对齐到了同一个维度上
    Q_s = quantile(Ys, prob_axis);
    Q_t = quantile(Yt, prob_axis);
    
    % 4. 计算距离 (L1 距离)
    % 物理含义：将分布 S 搬运变为分布 T 所需移动的平均"土量"
    dist = mean(abs(Q_s - Q_t));
end
