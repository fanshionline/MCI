import numpy as np


'''
变分模态分解(Variational Model Decomposition, VMD)算法
Input:
    signal: 要分解的1维时域信号
    alpha: 带宽限制
    tau: 噪声耐受，一般取0
    K: 分解出的模态数量
    DC: 如果为True则分解出的模态中包含直流分量
    init: 0 = 中心频率初始化为0
          1 = 中心频率按均匀分布初始化
          2 = 中心频率随机初始化
    tol: 收敛边界，一般取1e-6
Output:
    u: 每个模态的时域信号
    u_hat: 每个模态的波谱
    omega: 计算出各模态的中心频率
'''

k_value = None

def VMD(signal, alpha, tau, K, DC, init, tol):
    import math
    # 周期与采样频率
    save_T=len(signal)
    fs=1/float(save_T)

    # 使用镜像对称将信号扩展为2倍宽
    T=save_T
    f_mirror=np.zeros(2*T)
    f_mirror[0:T//2]=signal[-T//2-1::-1]
    f_mirror[T//2:3*T//2]= signal
    f_mirror[3*T//2:2*T]=signal[-1:-T//2-1:-1]
    f=f_mirror

    # 镜像信号的时域从，0到T
    T=float(len(f))
    t=np.linspace(1/float(T),1,int(T),endpoint=True)

    # 频谱离散化
    freqs=t-0.5-1/T
    # 最大迭代次数
    N=500

    # 每个模式都有自己的alpha
    Alpha=alpha*np.ones(K,dtype=complex)

    # 初始化模态变量
    f_hat=np.fft.fftshift(np.fft.fft(f))
    f_hat_plus=f_hat
    f_hat_plus[0:int(T/2)]=0
    # 记录每一次迭代的信号模态
    u_hat_plus=np.zeros((N,len(freqs),K),dtype=complex)


    # 初始化中心频率
    omega_plus=np.zeros((N,K),dtype=complex)
                        
    if (init==1):
        for i in range(1,K+1):
            omega_plus[0,i-1]=(0.5/K)*(i-1)  # 均匀分布
    elif (init==2):
        omega_plus[0,:]=np.sort(math.exp(math.log(fs))+(math.log(0.5)-math.log(fs))*np.random.rand(1,K))  # 随机
    else:
        omega_plus[0,:]=0  # 全部置0

    if (DC):
        omega_plus[0,0]=0  # 如果分解的模态中包含直流，则第一个模态就是直流


    # 初始化双重变量
    lamda_hat=np.zeros((N,len(freqs)),dtype=complex)

    uDiff=tol+2.2204e-16  # 更新步长
    n = 1 # 循环计数器
    sum_uk=0 # 累加

    T=int(T)


    # 开始VMD算法的循环迭代过程
    while uDiff > tol and n<N:
        # 第一个模态的累加器
        k = 1
        sum_uk = u_hat_plus[n-1,:,K-1]+sum_uk-u_hat_plus[n-1,:,0]
        # 通过残差维纳滤波更新第一个模态的频谱
        u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
        if DC==False:
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T,k-1])))


        for k in range(2,K+1):
            # 累加
            sum_uk=u_hat_plus[n,:,k-2]+sum_uk-u_hat_plus[n-1,:,k-1]

            # 计算模态频谱
            u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
            
            # 中心频率
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T:,k-1])))
        # 更新双重变量
        lamda_hat[n,:]=lamda_hat[n-1,:]+tau*(np.sum(u_hat_plus[n,:,:],axis=1)-f_hat_plus)

        # 记录循环次数
        n = n + 1

        # 计算与原信号的差值，用于判断是否已经达到循环终止条件
        uDiff=2.2204e-16
        for i in range(1,K+1):
            uDiff=uDiff+1/float(T)*np.dot(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1],(np.conj(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1])).conj().T) 
        uDiff=np.abs(uDiff)
        
    # 后处理

    # 如果循环过早停止，去掉空位置
    N=np.minimum(N,n)
    omega = omega_plus[0:N,:]

    # 重构模态的信号
    u_hat = np.zeros((T,K),dtype=complex)
    u_hat[T//2:T,:]= np.squeeze(u_hat_plus[N-1,T//2:T,:])
    u_hat[T//2:0:-1,:]=np.squeeze(np.conj(u_hat_plus[N-1,T//2:T,:]))
    u_hat[0,:]=np.conj(u_hat[-1,:])
    u=np.zeros((K,len(t)),dtype=complex)
    for k in range(1,K+1):
        u[k-1,:]= np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k-1])))

    # 去除镜像的部分
    u=u[:,T//4:3*T//4]

    # 重新计算各模态信号的频谱
    u_hat = np.zeros((T//2,K),dtype=complex)

    for k in range(1,K+1):
        u_hat[:,k-1]=np.fft.fftshift(np.fft.fft(u[k-1,:])).conj().T

    return (u,u_hat,omega)


'''
调用VMD算法提取输入信号中的心率与呼吸值
Input:
    HRsignal: 需要提取呼吸心率的雷达信号
Output:
    measuredHeartbeat: 计算出的心率（次/分钟）
    measuredbreath: 计算出的心率（次/分钟）
'''
def VMD_hb(Signal, iK):
    # VMD算法需要使用的参数
    alpha = 2000  # 带宽限制
    tau = 0  # 噪声耐受，一般取0
    K = iK  # 分解出的模态数量
    DC = 0  # 结果不包含直流部分
    init = 1  # 初始化omegas(中心频率)为均匀分布
    tol = 1e-7  # 总信号拟合误差允许程度
    DData = Signal  # 待分解的信号

    [u, u_hat, omega] = VMD(DData, alpha, tau, K, DC, init, tol)  # 调用VMD算法

    # 按照中心频率从小到大排列
    sortIndex = np.argsort(omega[len(omega)-1, :])
    omega = omega[:, sortIndex]
    u_hat = u_hat[:, sortIndex]
    u = u[sortIndex, :]
    return u


'''
提取输入信号中的心率与呼吸值
Input:
    HRsignal: 需要提取呼吸心率的雷达信号
Output:
    measuredHeartbeat: 计算出的心率（次/分钟）
    measuredbreath: 计算出的心率（次/分钟）
'''
def VMDHRBR(HRsignal):
    global k_value
    # 低带3Hz
    hr_filter = [0.0309909673242566, -0.165661939691912, 0.0108793969808390, 0.0324236783132672,
                 0.0199968912456759, -0.000613578794395767, -0.0174981843140624, -0.0209931469194497,
                 -0.00877704765628144, 0.0108700045894203, 0.0230147108320965, 0.0169173382647123,
                 -0.00452647999755323, -0.0249793770879194, -0.0263563796709082, -0.00399704934895674,
                 0.0266844715479274, 0.0395500004882902, 0.0181203159485759, -0.0279207261245773, -0.0638278713762797,
                 -0.0504032960259331, 0.0287131587511762, 0.150153024415508, 0.260050540027963, 0.304523197983351,
                 0.260050540027963, 0.150153024415508, 0.0287131587511762, -0.0504032960259331, -0.0638278713762797,
                 -0.0279207261245773, 0.0181203159485759, 0.0395500004882902, 0.0266844715479274, -0.00399704934895674,
                 -0.0263563796709082, -0.0249793770879194, -0.00452647999755323, 0.0169173382647123, 0.0230147108320965,
                 0.0108700045894203, -0.00877704765628144, -0.0209931469194497, -0.0174981843140624,
                 -0.000613578794395767, 0.0199968912456759, 0.0324236783132672, 0.0108793969808390, -0.165661939691912,
                 0.0309909673242566]
    valueofk = [4,7,10,12,15,20]  # VMD算法提取模态数量K，若分解不出心率与呼吸速率对应频率段信号则增大K继续计算
    # valueofk = [7]
    # 在Python中，“**”表示指数，“//”表示输出结果为除法向下取整
    Y = range(-2 ** 16 // 2, 2 ** 16 // 2 - 1)  # 16位整型数的范围 [-32768, 32767]
    Y = np.array(Y) * 20 / 2 ** 16  # [-10, 10 - 20 / 2** 16]
    flagheart = 0  # 计算出心率结果则置1
    flagbreath = 0  # 计算出呼吸速率结果则置1
    DData = np.convolve(HRsignal, hr_filter)[25:len(HRsignal)+25]
    measuredHeartbeat = -1  # 计算出的心率
    measuredbreath = -1  # 计算出的呼吸速率

    for j in range(len(valueofk)):  # 经过几次VMD分解由valueofk中的K值个数来定，若分解不出心率与呼吸速率对应频率段信号则增大K继续计算

        if flagheart == 0 or flagbreath==0:  # 没有找到心率和呼吸则继续增大K进行VMD分解
            K = valueofk[j]
            u = VMD_hb(DData, K)  # 以当前K个模态准备调用VMD算法分解

            fre = np.zeros((K, 2**16))  # 计算频率的矩阵
            provalue = np.zeros((1, K))  # 存储分解出的K个模态的中心频率值
            print('k:', K)
            k_value = K
            for i in range(K):
                spectrum = np.fft.fft(u[i, :], 2 ** 16)
                fre[i, :] = np.abs(np.fft.fftshift(spectrum)) * 10000
                result = np.argmax(fre[i,:])
                provalue[0,i] = int(np.abs(Y[result]) * 60)  # 存储所有分解出的频率值保存为次/分钟单位
                # print(provalue[0,i])

            if flagheart==0 and flagbreath==0:
                for i in range(K):
                    if 50 <= provalue[0,i] <= 110:  # 如果有中心频率在心率的范围内
                        measuredHeartbeat = provalue[0,i]  # 心率
                        flagheart = 1  # 分解出了心率
                    if 10 <= provalue[0,i] <= 27:  # 如果有中心频率在呼吸频率的范围内
                        measuredbreath = provalue[0,i]  # 呼吸频率
                        flagbreath = 1  # 分解出了呼吸频率
            
                if flagheart and flagbreath:
                    pass
                else:
                    flagheart = 0
                    flagbreath = 0
                    
        else:
            break

    if (flagheart == 0):
        measuredHeartbeat = 80  # 如果没有分解出心率，则心率值置80
    if (flagbreath == 0):
        measuredbreath = 18  # 如果没有分解出呼吸频率，则呼吸频率值置18

    return measuredHeartbeat, measuredbreath
