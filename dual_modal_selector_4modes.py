#!/usr/bin/env python

import pyvisa as visa
import numpy as np
import csv
import time

def load_waveform_with_time(filename):
    """讀取波形文件並返回時間和數值數據"""
    times = []
    values = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for t, p in reader:
            times.append(float(t))
            values.append(float(p))
    
    return np.array(times), np.array(values)

def align_waveforms(file1, file2, invert_ch2=True):
    """讀取兩個波形檔案並對齊時間軸"""
    times1, values1 = load_waveform_with_time(file1)
    times2, values2 = load_waveform_with_time(file2)
    
    # 找到共同的時間範圍
    t_start = max(times1[0], times2[0])
    t_end = min(times1[-1], times2[-1])
    
    # 計算統一的採樣率（使用較高的採樣率）
    dt1 = np.mean(np.diff(times1))
    dt2 = np.mean(np.diff(times2))
    dt_unified = min(dt1, dt2)  # 使用較小的時間間隔（較高採樣率）
    
    # 創建統一的時間軸
    unified_times = np.arange(t_start, t_end + dt_unified, dt_unified)
    
    # 插值到統一時間軸
    aligned_values1 = np.interp(unified_times, times1, values1)
    aligned_values2 = np.interp(unified_times, times2, values2)
    
    # 對 Channel 2 進行反相
    if invert_ch2:
        aligned_values2 = -aligned_values2
    
    # 計算採樣率
    sRate = str(1 / dt_unified)
    
    return (aligned_values1.astype('f4'), aligned_values2.astype('f4'), 
            sRate, len(unified_times), unified_times)

def setup_sync_internal(inst):
    """設定 Sync Internal (Track On) 功能"""
    # 先確保兩個通道都關閉追蹤
    inst.write('SOUR1:TRACK OFF')
    inst.write('SOUR2:TRACK OFF')
    
    # 讓 Channel 2 追蹤 Channel 1 (這就是 Sync Internal 的核心)
    inst.write('SOUR2:TRACK ON')
    
    inst.write('*WAI')

def run_mode(inst, mode_num):
    """執行指定模式的波形輸出"""
    start_time = time.time()
    
    # 統一電壓設定
    ch1_voltage = 1.2
    ch2_voltage = 1.2
    
    # 根據模式選擇波形檔案和極性設定
    if mode_num == 1:
        file1 = 'modal/25k_50k_84p88deg_2000pts.dat'
        file2 = 'modal/25k_50k_264p88deg_2000pts.dat'
        ch1_polarity = 'NORM'
        ch2_polarity = 'INV'
    elif mode_num == 2:
        file1 = 'modal/47k_94k_57p32deg_2000pts.dat'
        file2 = 'modal/47k_94k_237p32deg_2000pts.dat'
        ch1_polarity = 'NORM'
        ch2_polarity = 'INV'
    elif mode_num == 3:
        file1 = 'modal/25k_50k_84p88deg_2000pts.dat'
        file2 = 'modal/25k_50k_264p88deg_2000pts.dat'
        ch1_polarity = 'INV'
        ch2_polarity = 'NORM'
    else:  # mode_num == 4
        file1 = 'modal/47k_94k_57p32deg_2000pts.dat'
        file2 = 'modal/47k_94k_237p32deg_2000pts.dat'
        ch1_polarity = 'INV'
        ch2_polarity = 'NORM'
    
    # 先關閉輸出
    inst.write('OUTP1 OFF')
    inst.write('OUTP2 OFF')
    inst.write('SOUR2:TRACK OFF')
    inst.write('*WAI')
    
    # 讀取並對齊波形
    sig1, sig2, sRate, points, unified_times = align_waveforms(file1, file2, invert_ch2=True)
    
    # 上傳波形
    inst.write('SOUR1:DATA:VOL:CLE')
    inst.write_binary_values('SOUR1:DATA:ARB MODAL_84DEG,', sig1, datatype='f', is_big_endian=False)
    inst.write('*WAI')
    inst.write('MMEM:STOR:DATA "INT:\\remoteAdded\\MODAL_84DEG.arb"')
    
    inst.write('SOUR2:DATA:VOL:CLE')
    inst.write_binary_values('SOUR2:DATA:ARB MODAL_264DEG,', sig2, datatype='f', is_big_endian=False)
    inst.write('*WAI')
    inst.write('MMEM:STOR:DATA "INT:\\remoteAdded\\MODAL_264DEG.arb"')
    
    # 配置參數
    inst.write('SOUR1:FUNC ARB')
    inst.write('SOUR1:FUNC:ARB MODAL_84DEG')
    inst.write('SOUR1:FUNC:ARB:SRAT ' + sRate)
    inst.write(f'SOUR1:VOLT {ch1_voltage}')
    inst.write('SOUR1:VOLT:OFFS 0')
    
    inst.write('SOUR2:FUNC ARB')
    inst.write('SOUR2:FUNC:ARB MODAL_264DEG')
    inst.write('SOUR2:FUNC:ARB:SRAT ' + sRate)
    inst.write(f'SOUR2:VOLT {ch2_voltage}')
    inst.write('SOUR2:VOLT:OFFS 0')
    
    freq = float(sRate) / points
    inst.write(f'SOUR1:FREQ {freq}')
    inst.write(f'SOUR2:FREQ {freq}')
    inst.write('SOUR1:PHAS 0')
    inst.write('SOUR2:PHAS 0')
    inst.write('*WAI')
    
    setup_sync_internal(inst)
    
    inst.write('SOUR2:PHAS:SYNC')
    inst.write('SOUR2:PHAS 0')
    inst.write('*WAI')
    
    inst.write(f'OUTP1:POL {ch1_polarity}')
    inst.write(f'OUTP2:POL {ch2_polarity}')
    
    inst.write('OUTP:SYNC ON')
    inst.write('OUTP:SYNC:SOURCE CH1')
    inst.write('OUTP:SYNC:MODE MARK')
    
    inst.write('OUTP1 ON')
    inst.write('OUTP2 ON')
    inst.write('*WAI')
    
    inst.write("DISP:TEXT ''")
    
    switch_time = (time.time() - start_time) * 1000
    
    print(f"Mode {mode_num} | {switch_time:.1f}ms")
    
    return freq

if __name__ == "__main__":
    print("Dual Modal Controller")
    print("Connecting...")
    
    rm = visa.ResourceManager()
    inst = rm.open_resource('USB0::0x0957::0x5707::MY59001615::0::INSTR')
    
    try:
        inst.control_ren(6)
    except:
        pass
    
    # 溫和的初始化
    print("Initializing...")
    inst.write('OUTP1 OFF')
    inst.write('OUTP2 OFF')
    inst.write('SOUR1:TRACK OFF')
    inst.write('SOUR2:TRACK OFF')
    inst.write('*WAI')
    
    inst.write('*CLS')
    inst.write('*WAI')
    
    inst.write("MMEMORY:MDIR \"INT:\\remoteAdded\"")
    inst.write('FORM:BORD SWAP')
    
    # 初始模式說明
    print("\nModes:")
    print("Mode 1: 25k-50k Hz, CH1:Normal, CH2:Inverted")
    print("Mode 2: 47k-94k Hz, CH1:Normal, CH2:Inverted") 
    print("Mode 3: 25k-50k Hz, CH1:Inverted, CH2:Normal")
    print("Mode 4: 47k-94k Hz, CH1:Inverted, CH2:Normal")
    print("Ready")
    
    while True:
        try:
            print("\n1-Mode1  2-Mode2  3-Mode3  4-Mode4  q-Quit")
            
            user_input = input("Select: ").strip().lower()
            
            if user_input in ['1', '2', '3', '4']:
                mode_num = int(user_input)
                freq = run_mode(inst, mode_num)
                
            elif user_input == 'q':
                inst.write('OUTP1 OFF')
                inst.write('OUTP2 OFF')
                inst.write('SOUR2:TRACK OFF')
                inst.close()
                break
                
            else:
                print("Invalid")
                
        except KeyboardInterrupt:
            inst.write('OUTP1 OFF')
            inst.write('OUTP2 OFF')
            inst.write('SOUR2:TRACK OFF')
            inst.close()
            break
        except Exception as e:
            print(f"Error: {e}")