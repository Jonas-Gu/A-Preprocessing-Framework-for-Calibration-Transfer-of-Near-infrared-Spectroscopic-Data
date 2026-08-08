# centering_funcs.py
import numpy as np

def none(X_train, X_val=None):
    return X_train, X_val

def mean_centering(X_train, X_val=None):
    """均值中心化: Val 使用 Train 的均值"""
    # 1. 计算训练集均值
    mean_vec = np.mean(X_train, axis=0)
    
    # 2. 训练集中心化
    X_train_new = X_train - mean_vec
    
    # 3. 验证集中心化
    X_val_new = None
    if X_val is not None:
        X_val_new = X_val - mean_vec
        
    return X_train_new, X_val_new

#methods = [none]
methods = [none, mean_centering]