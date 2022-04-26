import numpy as np


'''
求序列中目标簇数量
Input:
    signal: 原始信号
    N: 目标簇数量
    R: 簇半径
    max_iteration: 最大寻找次数，如果寻找了max_iteration次也没有找全N个目标簇，则退出
Output:
    amplitude: N个目标簇的最大幅度
    index: 每个最大幅度在原始信号中对应的序号
    energy_ratio: 第k个信号簇在去除前k-1个簇后的能量中所占能量比重，用于后续判别是否是反射或者多径
'''
def sp(signal:np.ndarray, N=10, R=15, max_iteration=30):
    amplitude = []
    index = []
    energy_ratio = []
    signal_processed = signal.copy()
    n = 0
    iteration = 0
    while n < N:
        i = np.argmax(signal_processed)
        signal_cluster = signal[i - R if i - R >=0 else 0 : i + R]
        iteration += 1
        if np.argwhere((signal[i] >= signal_cluster) == False).shape[0] == 0:
            amplitude.append(signal[i])
            index.append(i)
            energy_ratio.append(np.sum(signal_processed[i - R if i - R >=0 else 0 : i + R] * signal_processed[i - R if i - R >=0 else 0 : i + R]) / np.sum(signal_processed * signal_processed))
            n += 1
        signal_processed[i - R if i - R >=0 else 0 : i + R] = 0
        if iteration >= max_iteration:
            break
    return np.array(amplitude), np.array(index), np.array(energy_ratio)


'''
将信号序列（矩阵）去平均
Input:
    signal: 原始信号
Output:
    signal_subtration: 去除平均值后的信号
'''
def remove_average(signal:np.ndarray):
    return signal - np.mean(signal, 0)


'''
计算信号序列（矩阵）的能量平均值
Input:
    signal: 原始信号
Output:
    signal_energy: 信号的平均能量
'''
def signal_energy(signal:np.ndarray):
    return np.mean(signal * signal, 0)