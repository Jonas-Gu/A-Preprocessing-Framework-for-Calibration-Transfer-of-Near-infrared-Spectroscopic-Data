# scatter_funcs.py
import numpy as np

def none(X_train, X_val=None):
    return X_train, X_val

def snv(X_train, X_val=None):
    def apply_snv(X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1e-10
        return (X - mean) / std

    X_train_new = apply_snv(X_train)
    X_val_new = None
    if X_val is not None:
        X_val_new = apply_snv(X_val)
    return X_train_new, X_val_new

def msc(X_train, X_val=None):
    # 1. 计算训练集的参考光谱
    mean_spectrum = np.mean(X_train, axis=0)
    
    def apply_msc_logic(X, ref):
        n_samples = X.shape[0]
        X_msc = np.zeros_like(X)
        for i in range(n_samples):
            # Fit: Sample ~ Reference
            fit = np.polyfit(ref, X[i, :], 1)
            slope = fit[0]
            intercept = fit[1]
            X_msc[i, :] = (X[i, :] - intercept) / slope
        return X_msc

    # 2. 应用到训练集
    X_train_new = apply_msc_logic(X_train, mean_spectrum)
    
    # 3. 应用到验证集 (使用同一个 mean_spectrum)
    X_val_new = None
    if X_val is not None:
        X_val_new = apply_msc_logic(X_val, mean_spectrum)
        
    return X_train_new, X_val_new

def emsc(X_train, X_val=None):
    """
    扩展多元散射校正 (EMSC) - 2阶多项式版本
    拟合模型: X ~ b*Ref + c0 + c1*w + c2*w^2
    校正公式: X_new = (X - c0 - c1*w - c2*w^2) / b
    """
    # 1. 计算参考光谱 (训练集均值)
    mean_spectrum = np.mean(X_train, axis=0)
    
    # 2. 构建波长向量
    n_features = X_train.shape[1]
    w = np.arange(n_features, dtype=float)
    
    # 3. 构建设计矩阵 M = [Ref, 1, w, w^2]
    # 这是一个多元线性回归: Sample = M * Coeffs
    # M 的形状: (n_features, 4)
    M = np.vstack((mean_spectrum, np.ones(n_features), w, w**2)).T
    
    def apply_emsc_logic(X):
        n_samples = X.shape[0]
        X_emsc = np.zeros_like(X)
        
        for i in range(n_samples):
            y = X[i, :] # 当前样本
            
            # 多元线性回归求解: y = M * c
            # c[0] = b (参考谱系数/缩放因子)
            # c[1] = c0 (常数基线)
            # c[2] = c1 (线性基线)
            # c[3] = c2 (二次基线)
            coeffs, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
            
            b = coeffs[0]
            baseline = coeffs[1] + coeffs[2]*w + coeffs[3]*(w**2)
            
            # 校正: 去除基线，归一化缩放
            # 避免 b 为 0
            if b == 0: b = 1e-10
            
            X_emsc[i, :] = (y - baseline) / b
            
        return X_emsc

    # 应用到训练集
    X_train_new = apply_emsc_logic(X_train)
    
    # 应用到验证集 (这里的M矩阵包含了mean_spectrum，也就是包含了训练集的信息)
    X_val_new = None
    if X_val is not None:
        X_val_new = apply_emsc_logic(X_val)
        
    return X_train_new, X_val_new
#methods = [none]
methods = [none, snv, msc,emsc]
