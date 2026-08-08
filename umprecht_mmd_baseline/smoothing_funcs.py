# smoothing_funcs.py
import numpy as np
from scipy.signal import savgol_filter
from utils_config import get_default_window

def none(X_train, X_val=None):
    return X_train, X_val

def sgolay(X_train, X_val=None):
    """Savitzky-Golay 平滑"""
    window = get_default_window()
    if window % 2 == 0:
        window += 1
    # polyorder=2 对应 MATLAB 默认设置
    X_train_new = savgol_filter(X_train, window_length=window, polyorder=2, axis=1)
    
    X_val_new = None
    if X_val is not None:
        X_val_new = savgol_filter(X_val, window_length=window, polyorder=2, axis=1)
        
    return X_train_new, X_val_new

def moving_average(X_train, X_val=None):
    """移动平均平滑"""
    window = get_default_window()
    kernel = np.ones(window) / window
    
    def apply_ma(X):
        # mode='same' 保持输出尺寸一致
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=X)

    X_train_new = apply_ma(X_train)
    
    X_val_new = None
    if X_val is not None:
        X_val_new = apply_ma(X_val)
        
    return X_train_new, X_val_new

#methods = [none]
methods = [none, sgolay, moving_average]