# baseline_funcs.py
import numpy as np
from scipy.signal import savgol_filter
from utils_config import get_default_window

# ================= 辅助函数 =================
def _apply_savgol(X, polyorder, deriv):
    """
    通用Savitzky-Golay包装器
    """
    window = get_default_window()
    # Scipy 要求窗口必须是奇数
    if window % 2 == 0: 
        window += 1
    
    # 注意：polyorder 必须小于 window_length
    if polyorder >= window:
        polyorder = window - 1
        
    return savgol_filter(X, window_length=window, polyorder=polyorder, deriv=deriv, axis=1)

def _wrap_func(X_train, X_val, polyorder, deriv):
    """
    处理 Train 和 Val 的双输入逻辑
    """
    X_train_new = _apply_savgol(X_train, polyorder, deriv)
    
    X_val_new = None
    if X_val is not None:
        X_val_new = _apply_savgol(X_val, polyorder, deriv)
        
    return X_train_new, X_val_new

# ================= 具体方法实现 =================
def none(X_train, X_val=None):
    return X_train, X_val

def derivate1_1(X_train, X_val=None):
    """1st Deriv, Poly Order 1"""
    return _wrap_func(X_train, X_val, polyorder=1, deriv=1)

def derivate1_3(X_train, X_val=None):
    """1st Deriv, Poly Order 3"""
    return _wrap_func(X_train, X_val, polyorder=3, deriv=1)

def derivate1_4(X_train, X_val=None):
    """1st Deriv, Poly Order 4"""
    return _wrap_func(X_train, X_val, polyorder=4, deriv=1)

# --- 二阶导数 (Derivative 2) ---
def derivate2_2(X_train, X_val=None):
    """2nd Deriv, Poly Order 2"""
    return _wrap_func(X_train, X_val, polyorder=2, deriv=2)

def derivate2_3(X_train, X_val=None):
    """2nd Deriv, Poly Order 3"""
    return _wrap_func(X_train, X_val, polyorder=3, deriv=2)

def derivate2_4(X_train, X_val=None):
    """2nd Deriv, Poly Order 4"""
    return _wrap_func(X_train, X_val, polyorder=4, deriv=2)

def derivate2_5(X_train, X_val=None):
    """2nd Deriv, Poly Order 5"""
    return _wrap_func(X_train, X_val, polyorder=5, deriv=2)

# ================= 方法列表 =================
# 包含 none, detrend 以及所有的导数变体
methods = [none,derivate1_1,derivate1_3,derivate1_4,derivate2_2,derivate2_3,derivate2_4,derivate2_5]
