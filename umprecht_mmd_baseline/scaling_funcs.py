# scaling_funcs.py
import numpy as np

def none(X_train, X_val=None):
    return X_train, X_val

def auto_scaling(X_train, X_val=None):
    """Auto Scaling: Val 使用 Train 的标准差"""
    # 1. 计算训练集标准差
    std_vec = np.std(X_train, axis=0)
    std_vec[std_vec < 1e-10] = 1.0
    
    # 2. 应用
    X_train_new = X_train / std_vec
    
    X_val_new = None
    if X_val is not None:
        X_val_new = X_val / std_vec
        
    return X_train_new, X_val_new

methods = [none, auto_scaling]

