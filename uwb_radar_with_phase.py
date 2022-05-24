# created by Xikang Jiang
from pymoduleconnector import ModuleConnector
import logging
import numpy as np
import sys
from VMDHRBR import VMDHRBR
import pca_filter
import threading
from time import sleep
import time
import os
import scipy.io as sio


class uwb_radar_with_phase:
    leiji = None
    phase = None
    pure_data = None
    pure_phase_data = None
    xep = None
    can_read = False
    read_condition = None
    end = False
    list_length = 0
    range = [0, 0]
    now_read = None

    def __init__(self, com, kwargs):
        self.init_radar(com, kwargs)
        print('End init')

    def init_radar(self, com, kwargs):
        self.range = kwargs['set_frame_area']
        self.mc = ModuleConnector(com, log_level=0)
        self.xep = self.mc.get_xep()

        # inti x4driver
        self.xep.x4driver_init()
        # print('initing')

        # Set enable pin
        self.xep.x4driver_set_enable(kwargs['set_enable'])

        # Set iterations
        self.xep.x4driver_set_iterations(kwargs['set_iterations'])
        # Set pulses per step
        self.xep.x4driver_set_pulses_per_step(kwargs['set_pulses_per_step'])
        # Set dac step
        self.xep.x4driver_set_dac_step(kwargs['set_dac_step'])
        # Set dac min
        self.xep.x4driver_set_dac_min(kwargs['set_dac_min'])
        # Set dac max
        self.xep.x4driver_set_dac_max(kwargs['set_dac_max'])
        # Set TX power
        self.xep.x4driver_set_tx_power(kwargs['set_tx_power'])

        # Enable downconversion
        self.xep.x4driver_set_downconversion(kwargs['set_downconversion'])

        # Set frame area offset
        self.xep.x4driver_set_frame_area_offset(kwargs['set_frame_area_offset'])
        offset = self.xep.x4driver_get_frame_area_offset()

        # Set frame area
        self.xep.x4driver_set_frame_area(self.range[0], self.range[1])
        frame_area = self.xep.x4driver_get_frame_area()

        # Set TX center freq
        self.xep.x4driver_set_tx_center_frequency(kwargs['set_tx_center_frequency'])

        # Set PRFdiv
        self.xep.x4driver_set_prf_div(kwargs['set_prf_div'])
        prf_div = self.xep.x4driver_get_prf_div()

        # Start streaming
        self.xep.x4driver_set_fps(kwargs['set_fps'])
        fps = self.xep.x4driver_get_fps()
        print("fps:", fps)
        print("wait radar")
        self.read_condition = threading.Condition()

        self.list_length = len(self.read_frame())

        self.read_frame_circulation = threading.Thread(
            name='read_frame_circulation', target=self.read_circulation, args=())
        self.preprocess = threading.Thread(
            name='preprocess', target=self.preprocessing, args=())

        self.read_frame_circulation.setDaemon(True)
        self.preprocess.setDaemon(True)

        self.read_frame_circulation.start()
        self.preprocess.start()


    def preprocessing(self):
        ycl = 1
        M = 20  # 输出结果去除前M行
        L = 50  # 输出结果去除前L列，同时后50列也会被去掉
        print('ready')
        while not self.end:
            sleep(0.1)
            if True:
            # with self.read_condition:
                # print(self.leiji.shape[0])
                if self.leiji is not None and self.leiji.shape[0] > 12*40:
                    if ycl == 1:
                        print('init')
                    self.leiji = self.leiji[-(11*40):, :]
                    self.phase = self.phase[-(11*40):, :]
                    raw_data = self.leiji[-(11*40):, :-1].copy()
                    raw_phase_data = self.phase[-(11*40):, :-1].copy()
                    self.pure_phase_data = pca_filter.p_f_phase(raw_phase_data)
                    self.pure_data = pca_filter.p_f(raw_data, M, L)
                    ycl += 1
                    self.can_read = True
                # self.read_condition.notify()

    def read_circulation(self):
        while not self.end:
            if True:
                save1, save2 = self.get_data()
                if self.phase is None:
                    self.leiji = save1.copy()
                    self.phase = save2.copy()
                else:
                    self.leiji = np.vstack((self.leiji, save1))
                    self.phase = np.vstack((self.phase, save2))
                self.now_read = save1.copy()
                # print('Read frame')
                # self.read_condition.notify()
                # print(time.time() - start)
            sleep(0.001)

    def read_frame(self):
        """Gets frame data from module"""
        d = self.xep.read_message_data_float()
        frame = np.array(d.data)
        # Convert the resulting frame to a complex array if downconversion is enabled
        return frame

    def get_data(self):
        save1 = np.ones((1, self.list_length))
        save2 = np.ones((1, self.list_length))
        save3 = np.ones((1, self.list_length))
        fc = 7.29e9
        fs = 23.328e9
        phase_filter = np.array(
            [-0.115974720052285, -0.0773345100048285, 0.0365253777771683, -0.0412732590195836, 0.0284568407504594,
             -0.0115161214947589, 0.0299994926179564, 0.00698978275147158, 0.0273822446913041, 0.00875491552837954,
             0.0131136766130344, -0.00510650646551428, -0.00946616689891689, -0.0241168278986621, -0.0274525374698975,
             -0.0322786742036804, -0.0261354557117109, -0.0167384782609816, 0.00194348360668676, 0.0240446970936053,
             0.0511262886037737, 0.0778124759461410, 0.103468064677834, 0.123475425164957, 0.137013410794562,
             0.141361791292646, 0.137013410794562, 0.123475425164957, 0.103468064677834, 0.0778124759461410,
             0.0511262886037737, 0.0240446970936053, 0.00194348360668676, -0.0167384782609816, -0.0261354557117109,
             -0.0322786742036804, -0.0274525374698975, -0.0241168278986621, -0.00946616689891689, -0.00510650646551428,
             0.0131136766130344, 0.00875491552837954, 0.0273822446913041, 0.00698978275147158, 0.0299994926179564,
             -0.0115161214947589, 0.0284568407504594, -0.0412732590195836, 0.0365253777771683, -0.0773345100048285,
             -0.115974720052285])

        if not self.end:
            
            frame2 = self.read_frame()
            csine = np.exp(-1j * fc / fs * 2 * np.pi * np.arange(frame2.shape[0]))
            cframe = frame2 * csine
            cframe_lp = np.convolve(phase_filter, cframe)[25:-25]
            phase = np.arctan(np.imag(cframe_lp) / np.real(cframe_lp))
            range = np.sqrt(np.square(np.imag(cframe_lp))+np.square(np.real(cframe_lp)))
            save1[0, :] = frame2
            save2[0, :] = phase
            save3[0, :] = range

        return save1, save2


    def max_energy(self, puredata):
        data = puredata
        energyOfLocation = [0] * data.shape[1]
        for j in range(data.shape[1]):
            energyOfLocation[j] = sum(data[:, j] * data[:, j])
        sig = puredata[:, np.argmax(energyOfLocation)]
        return sig

    def vital_signs(self, range_low, range_high):

        beginindex = round((range_low - self.range[0])*156 - 50)
        endindex = round((range_high - self.range[0])*156 - 50)
        beginindex = 0 if beginindex < 0 else beginindex
        endindex = 0 if endindex < 0 else endindex

        if self.can_read and self.pure_data is not None:

            beginindex = self.pure_data.shape[1] if beginindex>self.pure_data.shape[1] else beginindex
            endindex = self.pure_data.shape[1] if endindex>self.pure_data.shape[1] else endindex
            if beginindex == endindex:
                return 0, 0

            PureData = self.pure_data[:, int(beginindex):int(endindex)].copy()
            PurePhaseData = self.pure_phase_data[:, int(beginindex)+15:int(endindex)+15].copy()
            # 加15是预处理的补偿, 相位信号只去除了前35列而幅值信号去除了50列
            sig1 = self.max_energy(PureData)
            sig2 = self.max_energy(PurePhaseData)
            out = VMDHRBR(sig1)  # 使用经过预处理的幅值信号去进行VMD计算
            out_phase = VMDHRBR(sig2)  # 使用经过预处理的相位信号去进行VMD计算
            hr1 = int(round(out[0]))  # 幅值信号算出来的心率
            hr2 = int(round(out_phase[0]))  # 相位信号算出来的心率
            br1 = int(round(out[1]))  # 幅值信号算出来的呼吸速率
            br2 = int(round(out_phase[1]))  # 相位信号算出来的呼吸速率
            kval1 = int(out[2])  # 幅值信号算出来心率和呼吸速率时的k值
            kval2 = int(out_phase[2])  # 相位信号算出来心率和呼吸速率时的k值
            return hr1, br1, kval1, hr2, br2, kval2
        else:
            return 0, 0, 0, 0, 0, 0

    def reset(self):
        self.end = True
        if self.preprocess.is_alive():
            self.preprocess.join()
        if self.read_frame_circulation.is_alive():
            self.read_frame_circulation.join()
        self.xep.module_reset()
        print("radar reset!")
        return 1

    def __del__(self):
        print('destruction')
        if self.end != True:
            self.end = True
            if hasattr(self, 'read_frame_circulation'):
                if self.read_frame_circulation.is_alive():
                    self.read_frame_circulation.join()
            if hasattr(self, 'preprocess'):
                if self.preprocess.is_alive():
                    self.preprocess.join()
            self.xep.module_reset()


if __name__ == '__main__':
    # 清除之前积攒的debug文件
    files = os.listdir()
    for file in files:
        if not os.path.isdir(file):
            if file[0:5] == 'debug':
                os.remove(file)
    args = {
        'set_enable': 1,
        'set_iterations': 64,
        'set_pulses_per_step': 38,
        'set_dac_step': 1,
        'set_dac_min': 949,
        'set_dac_max': 1100,
        'set_tx_power': 2,
        'set_downconversion': 0,
        'set_frame_area_offset': 0.18,
        'set_frame_area': [0.2, 5],
        'set_tx_center_frequency': 3,
        'set_prf_div': 16,
        'set_fps': 40}
    uwb = uwb_radar_with_phase('COM3', args)
    i = 0
    while i < 40 * 30:
        hr = uwb.vital_signs(0, 1)[0]
        if hr != -1:
            print(hr)
        sleep(0.1)
        i += 1
        pass
    uwb.reset()
