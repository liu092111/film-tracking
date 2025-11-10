#!/usr/bin/env python

import pyvisa as visa
import numpy as np
import csv
import time

def load_waveform_csv(filename):
    """讀取 CSV 波形文件"""
    times = []
    values = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # 跳過空行和註解
                continue
            
            # 處理 CSV 格式 (逗號分隔)
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()  # 備用：空格分隔
            
            if len(parts) >= 2:
                try:
                    t = float(parts[0])
                    v = float(parts[1])
                    times.append(t)
                    values.append(v)
                except ValueError:
                    continue
    
    return np.array(times), np.array(values)

def align_waveforms(file1, file2, invert_ch2=True):
    """讀取並對齊波形 - 不進行任何標準化"""
    times1, values1 = load_waveform_csv(file1)
    times2, values2 = load_waveform_csv(file2)
    
    t_start = max(times1[0], times2[0])
    t_end = min(times1[-1], times2[-1])
    
    dt1 = np.mean(np.diff(times1))
    dt2 = np.mean(np.diff(times2))
    dt_unified = min(dt1, dt2)
    
    unified_times = np.arange(t_start, t_end + dt_unified, dt_unified)
    
    # 直接使用原始值，不做標準化
    aligned_values1 = np.interp(unified_times, times1, values1)
    aligned_values2 = np.interp(unified_times, times2, values2)
    
    if invert_ch2:
        aligned_values2 = -aligned_values2
    
    sRate = 1 / dt_unified
    
    return (aligned_values1.astype('f4'), aligned_values2.astype('f4'), 
            sRate, len(unified_times))

def clear_instrument_memory(inst):
    """清理儀器記憶體"""
    inst.write('SOUR1:DATA:VOL:CLE')
    inst.write('SOUR2:DATA:VOL:CLE')
    inst.write('*WAI')

def reset_instrument_completely(inst):
    """完全重置儀器狀態"""
    inst.write('OUTP1 OFF')
    inst.write('OUTP2 OFF')
    inst.write('SOUR1:TRACK OFF')
    inst.write('SOUR2:TRACK OFF')
    inst.write('*WAI')
    
    # 重置波形函數為預設
    inst.write('SOUR1:FUNC SIN')
    inst.write('SOUR2:FUNC SIN')
    inst.write('SOUR1:FREQ 1000')
    inst.write('SOUR2:FREQ 1000')
    inst.write('SOUR1:VOLT 0.1')
    inst.write('SOUR2:VOLT 0.1')
    inst.write('OUTP1:POL NORM')
    inst.write('OUTP2:POL NORM')
    inst.write('*WAI')
    
    # 清理自定義波形
    clear_instrument_memory(inst)

def preload_all_waveforms_continuous(inst):
    """預載入所有波形並設定常駐輸出 - 正確載入四個模式"""
    print("Loading all waveforms for continuous output...")
    clear_instrument_memory(inst)
    
    # 模式配置 - 使用正確的檔案名稱
    modes_config = {
        1: {
            'file1': 'modal/ONEPERIOD_A_25k_50k_84p88deg_2000pts.csv',
            'file2': 'modal/ONEPERIOD_B_25k_50k_264p88deg_2000pts.csv',
            'name1': 'WF_25K_84',
            'name2': 'WF_25K_264'
        },
        2: {
            'file1': 'modal/ONEPERIOD_C_47k_94k_57p32deg_2000pts.csv',
            'file2': 'modal/ONEPERIOD_D_47k_94k_237p32deg_2000pts.csv',
            'name1': 'WF_47K_57',
            'name2': 'WF_47K_237'
        },
        3: {
            'file1': 'modal/ONEPERIOD_A_25k_50k_84p88deg_2000pts.csv',
            'file2': 'modal/ONEPERIOD_B_25k_50k_264p88deg_2000pts.csv',
            'name1': 'WF_25K_84',  # 與 Mode 1 使用相同名稱
            'name2': 'WF_25K_264'  # 與 Mode 1 使用相同名稱
        },
        4: {
            'file1': 'modal/ONEPERIOD_C_47k_94k_57p32deg_2000pts.csv',
            'file2': 'modal/ONEPERIOD_D_47k_94k_237p32deg_2000pts.csv',
            'name1': 'WF_47K_57',  # 與 Mode 2 使用相同名稱
            'name2': 'WF_47K_237'  # 與 Mode 2 使用相同名稱
        }
    }
    
    # 關閉輸出進行安全的波形上傳
    inst.write('OUTP1 OFF')
    inst.write('OUTP2 OFF')
    inst.write('*WAI')
    
    sampling_rates = {}
    
    for mode_num, config in modes_config.items():
        print(f"Loading Mode {mode_num}...")
        
        try:
            # 不做標準化，保持原始數據點位置
            sig1, sig2, sRate, points = align_waveforms(
                config['file1'], config['file2'], invert_ch2=False  # 不在這裡反相
            )
            
            freq = sRate / points
            
            sampling_rates[mode_num] = {
                'sRate': sRate,
                'points': points,
                'freq': freq,
                'name1': config['name1'],
                'name2': config['name2']
            }
            
            # 上傳 Channel 1 波形
            print(f"  Uploading CH1: {config['name1']}")
            inst.write_binary_values(f'SOUR1:DATA:ARB {config["name1"]},', sig1, datatype='f', is_big_endian=False)
            inst.write('*WAI')
            
            # 上傳 Channel 2 波形 (原始數據，不預先反相)
            print(f"  Uploading CH2: {config['name2']}")
            inst.write_binary_values(f'SOUR2:DATA:ARB {config["name2"]},', sig2, datatype='f', is_big_endian=False)
            inst.write('*WAI')
            
            print(f"  Mode {mode_num} loaded - Freq: {freq:.1f} Hz (sRate: {sRate:.0f}, points: {points})")
            
        except Exception as e:
            print(f"  Error loading Mode {mode_num}: {e}")
    
    print("All waveforms loaded!")
    return sampling_rates

def setup_continuous_quad_output(inst):
    """設定四模式常駐輸出系統 - 所有模式同時載入，只透過振幅切換"""
    print("Setting up 4-mode continuous output system...")
    
    # 預載所有波形
    sampling_rates = preload_all_waveforms_continuous(inst)
    
    if not sampling_rates:
        print("ERROR: Failed to load waveforms!")
        return None, None
    
    # 設定常駐輸出 - 預設為Mode 1，其他模式振幅為0
    print("Setting up continuous output for all modes...")
    
    # 選擇使用 Mode 1 (25k) 作為基準輸出，因為它有最完整的參數
    base_mode = sampling_rates[1]
    
    # 設定基本輸出參數
    inst.write('SOUR1:FUNC ARB')
    inst.write('SOUR2:FUNC ARB')
    inst.write(f'SOUR1:FUNC:ARB {base_mode["name1"]}')
    inst.write(f'SOUR2:FUNC:ARB {base_mode["name2"]}')
    inst.write(f'SOUR1:FUNC:ARB:SRAT {base_mode["sRate"]:.0f}')
    inst.write(f'SOUR2:FUNC:ARB:SRAT {base_mode["sRate"]:.0f}')
    inst.write(f'SOUR1:FREQ {base_mode["freq"]}')
    inst.write(f'SOUR2:FREQ {base_mode["freq"]}')
    inst.write('SOUR1:PHAS 0')
    inst.write('SOUR2:PHAS 0')
    inst.write('SOUR1:VOLT:OFFS 0')
    inst.write('SOUR2:VOLT:OFFS 0')
    inst.write('*WAI')
    
    # 設定同步
    inst.write('SOUR1:TRACK OFF')
    inst.write('SOUR2:TRACK OFF')
    inst.write('SOUR2:TRACK ON')
    inst.write('SOUR2:PHAS:SYNC')
    inst.write('*WAI')
    
    # 初始設定：所有模式振幅為0 (待機狀態)
    inst.write('SOUR1:VOLT 0')
    inst.write('SOUR2:VOLT 0')
    # 不在這裡設定極性，讓各個模式自己設定
    inst.write('*WAI')
    
    # 開啟輸出並保持常駐
    inst.write('OUTP1 ON')
    inst.write('OUTP2 ON')
    inst.write('*WAI')
    
    print("Continuous quad-mode output ready!")
    print("All modes loaded, switching by amplitude/polarity only")
    
    return sampling_rates, None  # current_mode = None (待機)

def ultra_fast_amplitude_switch(inst, mode_num, current_mode, sampling_rates):
    """超快速振幅切換 - 只調整振幅和極性，不切換波形"""
    
    start_time = time.time()
    
    if mode_num not in sampling_rates:
        print(f"Mode {mode_num} not available!")
        return current_mode
    
    # 模式配置：只定義振幅和極性
    mode_configs = {
        1: {
            'ch1_amp': 1.2,      # CH1 活動
            'ch2_amp': 1.2,      # CH2 活動  
            'ch1_pol': 'NORM',   # CH1 正極性
            'ch2_pol': 'INV',    # CH2 反極性
            'wave_type': '25k',
            'desc': '25k Hz, CH1=NORM, CH2=INV'
        },
        2: {
            'ch1_amp': 1.2,      # CH1 活動
            'ch2_amp': 1.2,      # CH2 活動  
            'ch1_pol': 'NORM',   
            'ch2_pol': 'INV',    
            'wave_type': '47k',
            'desc': '47k Hz, CH1=NORM, CH2=INV'
        },
        3: {
            'ch1_amp': 1.2,      # CH1 活動
            'ch2_amp': 1.2,      # CH2 活動
            'ch1_pol': 'INV',    # CH1 反極性  
            'ch2_pol': 'NORM',   # CH2 正極性
            'wave_type': '25k',
            'desc': '25k Hz, CH1=INV, CH2=NORM'
        },
        4: {
            'ch1_amp': 1.2,      # CH1 活動
            'ch2_amp': 1.2,      # CH2 活動
            'ch1_pol': 'INV',    
            'ch2_pol': 'NORM',   
            'wave_type': '47k',
            'desc': '47k Hz, CH1=INV, CH2=NORM'
        }
    }
    
    config = mode_configs[mode_num]
    
    # 檢查是否需要切換波形組 (25k ↔ 47k)
    if current_mode is None:
        # 從待機狀態啟動
        need_waveform_switch = config['wave_type'] != '25k'  # 預設載入25k
    else:
        current_config = mode_configs[current_mode]
        need_waveform_switch = current_config['wave_type'] != config['wave_type']
    
    # 檢查是否需要極性變化
    need_polarity_change = False
    if current_mode is None:
        need_polarity_change = True  # 從待機狀態需要設定極性
    else:
        current_config = mode_configs[current_mode]
        need_polarity_change = (current_config['ch1_pol'] != config['ch1_pol'] or 
                              current_config['ch2_pol'] != config['ch2_pol'])
    
    # 只在需要時關閉輸出
    if need_waveform_switch or need_polarity_change:
        inst.write('OUTP1 OFF; OUTP2 OFF')
        
        if need_waveform_switch:
            # 切換波形、採樣率和頻率（確保頻率正確生效）
            mode_data = sampling_rates[mode_num]
            inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
            inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
            inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
            inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
            inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
            inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
            inst.write('SOUR2:PHAS:SYNC')
            inst.write('*WAI')
        
        if need_polarity_change:
            # 設定極性
            inst.write(f'OUTP1:POL {config["ch1_pol"]}')
            inst.write(f'OUTP2:POL {config["ch2_pol"]}')
            inst.write('*WAI')
        
        # 重新開啟輸出
        inst.write('OUTP1 ON; OUTP2 ON')
        inst.write('*WAI')
    
    # 總是設定振幅（這是最快的操作）
    inst.write(f'SOUR1:VOLT {config["ch1_amp"]}; SOUR2:VOLT {config["ch2_amp"]}')
    inst.write('*WAI')
    
    switch_time = (time.time() - start_time) * 1000
    
    print(f"Mode {mode_num} | {switch_time:.1f}ms")
    
    return mode_num

def setup_default_continuous_output(inst, mode_data):
    """設定預設連續輸出 (Mode 1)"""
    
    # 設定 Channel 1
    inst.write('SOUR1:FUNC ARB')
    inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
    inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
    inst.write('SOUR1:VOLT 1.2')
    inst.write('SOUR1:VOLT:OFFS 0')
    inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
    inst.write('SOUR1:PHAS 0')
    
    # 設定 Channel 2
    inst.write('SOUR2:FUNC ARB')
    inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
    inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
    inst.write('SOUR2:VOLT 1.2')
    inst.write('SOUR2:VOLT:OFFS 0')
    inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
    inst.write('SOUR2:PHAS 0')
    inst.write('*WAI')
    
    # 設定同步
    inst.write('SOUR1:TRACK OFF')
    inst.write('SOUR2:TRACK OFF')
    inst.write('SOUR2:TRACK ON')
    inst.write('SOUR2:PHAS:SYNC')
    inst.write('*WAI')
    
    # 設定 Mode 1 極性 (CH1=NORM, CH2=INV)
    inst.write('OUTP1:POL NORM')
    inst.write('OUTP2:POL INV')
    inst.write('*WAI')
    
    # 開啟輸出並保持常駐
    inst.write('OUTP1 ON')
    inst.write('OUTP2 ON')
    inst.write('*WAI')
    
    print(f"Default Mode 1 active - Freq: {mode_data['freq']:.1f} Hz, CH1=NORM, CH2=INV")
    
    return 'group_25k'


def ultra_fast_mode_switch_continuous(inst, mode_num, current_group, waveform_groups, group_data, sampling_rates):
    """超快速模式切換 - 正確的波形和極性設定"""
    
    start_time = time.time()
    
    if mode_num not in sampling_rates:
        print(f"Mode {mode_num} not available!")
        return current_group
        
    # 極性配置 - 確保正確的極性設定
    if mode_num == 1:
        ch1_polarity = 'NORM'
        ch2_polarity = 'INV'
        target_group = 'group_25k'
    elif mode_num == 2:
        ch1_polarity = 'NORM'
        ch2_polarity = 'INV'
        target_group = 'group_47k'
    elif mode_num == 3:
        ch1_polarity = 'INV'
        ch2_polarity = 'NORM'
        target_group = 'group_25k'
    else:  # mode_num == 4
        ch1_polarity = 'INV'
        ch2_polarity = 'NORM'
        target_group = 'group_47k'
    
    mode_data = sampling_rates[mode_num]
    
    # 情況1: 從待機狀態啟動 (完整設定)
    if current_group is None:
        # 完整設定並開啟輸出
        inst.write('SOUR1:FUNC ARB')
        inst.write('SOUR2:FUNC ARB')
        inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
        inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
        inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
        inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
        inst.write('SOUR1:VOLT 1.2')
        inst.write('SOUR2:VOLT 1.2')
        inst.write('SOUR1:VOLT:OFFS 0')  
        inst.write('SOUR2:VOLT:OFFS 0')
        inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
        inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
        inst.write('SOUR1:PHAS 0')
        inst.write('SOUR2:PHAS 0')
        inst.write('*WAI')
        
        # 設定同步
        inst.write('SOUR1:TRACK OFF')
        inst.write('SOUR2:TRACK OFF')
        inst.write('SOUR2:TRACK ON')
        inst.write('SOUR2:PHAS:SYNC')
        inst.write('*WAI')
        
        # 先設定極性，確保正確設定
        inst.write(f'OUTP1:POL {ch1_polarity}')
        inst.write(f'OUTP2:POL {ch2_polarity}')
        inst.write('*WAI')
        
        # 開啟輸出
        inst.write('OUTP1 ON')
        inst.write('OUTP2 ON') 
        inst.write('*WAI')
        
        switch_type = "完整啟動"
        
    # 情況2: 只需要改極性 (超快速 ~0.2ms)
    elif current_group == target_group:
        # 優化的極性切換 - 合併指令減少延遲
        inst.write('OUTP1 OFF; OUTP2 OFF')
        inst.write(f'OUTP1:POL {ch1_polarity}; OUTP2:POL {ch2_polarity}')
        inst.write('OUTP1 ON; OUTP2 ON')
        inst.write('*WAI')
        
        switch_type = "極性切換"
        
    # 情況3: 需要切換波形組 (中速 ~2ms) 
    else:
        # 關閉輸出進行安全切換
        inst.write('OUTP1 OFF')
        inst.write('OUTP2 OFF')
        inst.write('*WAI')
        
        # 設定完整的波形參數
        inst.write('SOUR1:FUNC ARB')
        inst.write('SOUR2:FUNC ARB')
        inst.write(f'SOUR1:FUNC:ARB {mode_data["name1"]}')
        inst.write(f'SOUR2:FUNC:ARB {mode_data["name2"]}')
        inst.write(f'SOUR1:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
        inst.write(f'SOUR2:FUNC:ARB:SRAT {mode_data["sRate"]:.0f}')
        inst.write('SOUR1:VOLT 1.2')
        inst.write('SOUR2:VOLT 1.2') 
        inst.write('SOUR1:VOLT:OFFS 0')
        inst.write('SOUR2:VOLT:OFFS 0')
        inst.write(f'SOUR1:FREQ {mode_data["freq"]}')
        inst.write(f'SOUR2:FREQ {mode_data["freq"]}')
        inst.write('SOUR1:PHAS 0')
        inst.write('SOUR2:PHAS 0')
        inst.write('*WAI')
        
        # 重新設定同步
        inst.write('SOUR1:TRACK OFF')
        inst.write('SOUR2:TRACK OFF')
        inst.write('SOUR2:TRACK ON')
        inst.write('SOUR2:PHAS:SYNC')
        inst.write('*WAI')
        
        # 先設定極性，確保優先生效
        inst.write(f'OUTP1:POL {ch1_polarity}')
        inst.write(f'OUTP2:POL {ch2_polarity}')
        inst.write('*WAI')
        
        # 重新開啟輸出
        inst.write('OUTP1 ON')
        inst.write('OUTP2 ON')
        inst.write('*WAI')
        
        switch_type = "波形+極性切換"
    
    switch_time = (time.time() - start_time) * 1000
    
    print(f"Mode {mode_num} | {switch_time:.2f}ms | {switch_type} | Freq: {mode_data['freq']:.1f}Hz | CH1={ch1_polarity}, CH2={ch2_polarity}")
    
    return target_group

if __name__ == "__main__":
    print("=== Ultra-Fast Dual Modal Controller ===")
    print("Features: <1ms polarity switching, ~2ms waveform switching")
    print("Connecting to instrument...")
    
    rm = visa.ResourceManager()
    inst = rm.open_resource('USB0::0x0957::0x5707::MY59001615::0::INSTR')
    
    try:
        inst.control_ren(6)
    except:
        pass
    
    # 完全重置儀器
    reset_instrument_completely(inst)
    
    # 初始化
    print("Initializing...")
    inst.write('*CLS')
    inst.write('*WAI')
    
    try:
        # 設定新的超快速振幅切換系統
        sampling_rates, current_mode = setup_continuous_quad_output(inst)
        
        if not sampling_rates:
            print("ERROR: System initialization failed!")
            reset_instrument_completely(inst)
            inst.close()
            exit(1)
        
        print("\n=== System Ready ===")
        print("Ultra-Fast Amplitude/Waveform Switching Mode")
        print("Mode 1: 25k waves, CH1=1.2V, CH2=1.2V, NORM/INV")
        print("Mode 2: 47k waves, CH1=1.2V, CH2=1.2V, NORM/INV")  
        print("Mode 3: 25k waves, CH1=1.2V, CH2=1.2V, INV/NORM")
        print("Mode 4: 47k waves, CH1=1.2V, CH2=1.2V, INV/NORM")
        print("純振幅切換 (同組): <0.5ms | 波形切換 (跨組): ~2-3ms")
        print("Current: Standby (amplitude=0) - select a mode to start")
        
        # 主控制迴圈
        while True:
            try:
                print(f"\nCurrent mode: {current_mode}")
                print("1-Mode1  2-Mode2  3-Mode3  4-Mode4  q-Quit")
                
                user_input = input("Select: ").strip().lower()
                
                if user_input in ['1', '2', '3', '4']:
                    mode_num = int(user_input)
                    current_mode = ultra_fast_amplitude_switch(
                        inst, mode_num, current_mode, sampling_rates
                    )
                
                elif user_input == 'q':
                    print("Cleaning up and exiting...")
                    reset_instrument_completely(inst)
                    inst.close()
                    print("Exit complete!")
                    break
                    
                else:
                    print("Invalid input!")
                    
            except KeyboardInterrupt:
                print("\nCleaning up and exiting...")
                reset_instrument_completely(inst)
                inst.close()
                print("Exit complete!")
                break
            except Exception as e:
                print(f"Error: {e}")
                reset_instrument_completely(inst)
                inst.close()
                break
                
    except Exception as e:
        print(f"Initialization error: {e}")
        reset_instrument_completely(inst)
        inst.close()
        exit(1)
