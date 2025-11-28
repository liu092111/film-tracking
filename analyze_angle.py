import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Find latest IMG_7129 folder
folders = [f for f in os.listdir('.') if os.path.isdir(f) and 'IMG_7129' in f]
latest = sorted(folders)[-1] if folders else None
print(f'Latest folder: {latest}')

if latest:
    # Get the correct CSV filename
    base_name = latest.replace('_analysis_output', '').replace('_output', '')
    csv_path = os.path.join(latest, f'{base_name}_pos_angle_speed.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f'CSV loaded: {csv_path}')
        print(f'Columns: {list(df.columns)}')
        
        print(f'\nAngle statistics:')
        print(df[['angle_deg_raw', 'angle_deg_unwrapped']].describe())
        
        # Plot angles
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df['t_s'], df['angle_deg_raw'], 'b.-', linewidth=0.5, markersize=2, label='Raw')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Raw Angle vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(df['t_s'], df['angle_deg_unwrapped'], 'r.-', linewidth=0.5, markersize=2, label='Unwrapped')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Unwrapped Angle vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        output_path = os.path.join(latest, 'angle_analysis.png')
        plt.savefig(output_path, dpi=150)
        print(f'\nAngle analysis plot saved to: {output_path}')
        
        # Analyze angle jumps
        angle_diff = np.abs(np.diff(df['angle_deg_raw'].dropna()))
        print(f'\nMax angle jump (raw): {np.max(angle_diff):.2f} degrees')
        print(f'Mean angle change: {np.mean(angle_diff):.2f} degrees')
        print(f'Number of large jumps (>10 deg): {np.sum(angle_diff > 10)}')
        print(f'Number of large jumps (>30 deg): {np.sum(angle_diff > 30)}')
        print(f'Number of large jumps (>60 deg): {np.sum(angle_diff > 60)}')
        
        # Show some examples of large jumps
        large_jump_indices = np.where(angle_diff > 30)[0]
        if len(large_jump_indices) > 0:
            print(f'\nFirst few large jumps (>30 deg):')
            for idx in large_jump_indices[:5]:
                print(f'  Frame {idx}: {df["angle_deg_raw"].iloc[idx]:.2f}° -> {df["angle_deg_raw"].iloc[idx+1]:.2f}° (jump: {angle_diff[idx]:.2f}°)')
    else:
        print(f'CSV file not found: {csv_path}')
else:
    print('No IMG_7129 folder found')
