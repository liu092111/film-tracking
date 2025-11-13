import os
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_line_and_deviation(x, y):
    """
    給一條軌跡 (x, y)，用最小平方法 fit 一條線 y = ax + b，
    並回傳每個點到這條線的垂直距離。
    """
    # 1st-order polyfit => y = a x + b
    a, b = np.polyfit(x, y, 1)

    # 垂直距離公式
    denom = math.sqrt(a * a + 1.0)
    d = np.abs(a * x - y + b) / denom

    return d, a, b


def process_single_csv(csv_path):
    """
    讀取單一 CSV，回傳偏差統計結果（如果欄位不符就回傳 None）。
    """
    df = pd.read_csv(csv_path)

    # 優先使用相對座標 x_mm, y_mm
    if {"x_mm", "y_mm"}.issubset(df.columns):
        x = df["x_mm"].to_numpy()
        y = df["y_mm"].to_numpy()
    # 備案：如果之後有只存 x_mm_abs, y_mm_abs 的檔案
    elif {"x_mm_abs", "y_mm_abs"}.issubset(df.columns):
        x_abs = df["x_mm_abs"].to_numpy()
        y_abs = df["y_mm_abs"].to_numpy()
        x = x_abs - x_abs[0]
        y = y_abs - y_abs[0]
    else:
        print(f"[skip] {csv_path} 缺少 x_mm / y_mm 欄位")
        return None

    # 去掉 NaN（例如最前面的速度沒有定義）
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        print(f"[skip] {csv_path} 有效點太少")
        return None

    d, a, b = fit_line_and_deviation(x, y)

    # 基本統計量
    mean_dev = float(np.mean(d))
    rms_dev = float(np.sqrt(np.mean(d ** 2)))
    max_dev = float(np.max(d))

    # 路徑長度（方便之後做 scatter: longer path vs 更歪？）
    seg_len = np.hypot(np.diff(x), np.diff(y))
    path_len = float(np.sum(seg_len))

    result = {
        "csv_file": os.path.basename(csv_path),
        "folder": os.path.basename(os.path.dirname(csv_path)),
        "mean_dev_mm": mean_dev,
        "rms_dev_mm": rms_dev,
        "max_dev_mm": max_dev,
        "path_length_mm": path_len,
    }

    # 如果有時間欄位，可以算平均速度
    if "t_s" in df.columns:
        t = df["t_s"].to_numpy()[mask]
        if len(t) > 1:
            duration = float(t[-1] - t[0])
            if duration > 0:
                result["duration_s"] = duration
                result["avg_speed_mm_s"] = path_len / duration

    return result


def collect_all_trials(root_dir="."):
    """
    掃描 repo，抓出所有名稱中含 'straight_integrated' 的資料夾，
    並處理裡面的所有 CSV。
    """
    records = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath)
        if "straight_integrated" not in folder_name:
            continue

        csv_files = glob.glob(os.path.join(dirpath, "*.csv"))
        for csv_path in csv_files:
            res = process_single_csv(csv_path)
            if res is not None:
                records.append(res)

    if not records:
        print("沒有找到任何符合條件的 CSV 檔案。")
        return None

    df = pd.DataFrame(records)
    return df


def plot_statistics(df, output_dir="straight_stats"):
    os.makedirs(output_dir, exist_ok=True)

    # 儲存 summary
    summary_path = os.path.join(output_dir, "straight_deviation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"已輸出統計結果: {summary_path}")

    # 1. 箱型圖（Mean / RMS / Max）—— 修正 tick_labels
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [df["mean_dev_mm"], df["rms_dev_mm"], df["max_dev_mm"]],
        tick_labels=["Mean dev", "RMS dev", "Max dev"],
    )
    plt.ylabel("Deviation (mm)")
    plt.title("Straightness Deviation (All Trials)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "boxplot_deviation.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"已輸出箱型圖: {boxplot_path}")

    # 2. RMS 偏差直方圖
    plt.figure(figsize=(8, 6))
    plt.hist(df["rms_dev_mm"], bins=10, edgecolor="black", alpha=0.7)
    plt.xlabel("RMS deviation (mm)")
    plt.ylabel("Count")
    plt.title("Histogram of RMS Deviation")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "hist_rms_deviation.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"已輸出直方圖: {hist_path}")

    # 3. RMS 偏差「區間長條圖」(0–2, 2–5, 5–10, 10+ mm)
    bins = [0, 2, 5, 10, np.inf]
    labels = ["0–2 mm", "2–5 mm", "5–10 mm", ">10 mm"]
    df["rms_bin"] = pd.cut(df["rms_dev_mm"], bins=bins, labels=labels, right=True)

    counts = df["rms_bin"].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("RMS deviation range (mm)")
    plt.ylabel("Number of trials")
    plt.title("RMS deviation grouped by range")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "bar_rms_range.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"已輸出 RMS 區間長條圖: {bar_path}")

    # 4. CDF of RMS deviation
    values = np.sort(df["rms_dev_mm"].to_numpy())
    cdf = np.arange(1, len(values) + 1) / len(values)

    plt.figure(figsize=(8, 6))
    plt.step(values, cdf, where="post")
    plt.xlabel("RMS deviation (mm)")
    plt.ylabel("Cumulative probability")
    plt.title("CDF of RMS deviation")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    # 若你有一個 spec，可以畫一條輔助線，例如 spec = 5 mm
    spec = 5.0
    plt.axvline(spec, color="gray", linestyle="--")
    plt.text(spec, 0.02, f"spec = {spec} mm", rotation=90, va="bottom", ha="right")
    cdf_path = os.path.join(output_dir, "cdf_rms_deviation.png")
    plt.savefig(cdf_path, dpi=300)
    plt.close()
    print(f"已輸出 CDF 圖: {cdf_path}")

    # 5. Path length vs RMS deviation（加 trend line）
    if "path_length_mm" in df.columns:
        x = df["path_length_mm"].to_numpy()
        y = df["rms_dev_mm"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y)

        # 線性回歸當趨勢線
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = np.polyval(coef, xs)
        plt.plot(xs, ys, linestyle="--")

        plt.xlabel("Path length (mm)")
        plt.ylabel("RMS deviation (mm)")
        plt.title("Path length vs RMS deviation")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, "scatter_length_vs_rms.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print(f"已輸出 Path length vs RMS 圖: {scatter_path}")

    # 6. Avg speed vs RMS deviation（如果有 avg_speed 欄位）
    if "avg_speed_mm_s" in df.columns:
        x = df["avg_speed_mm_s"].to_numpy()
        y = df["rms_dev_mm"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y)

        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = np.polyval(coef, xs)
        plt.plot(xs, ys, linestyle="--")

        plt.xlabel("Average speed (mm/s)")
        plt.ylabel("RMS deviation (mm)")
        plt.title("Average speed vs RMS deviation")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        scatter2_path = os.path.join(output_dir, "scatter_speed_vs_rms.png")
        plt.savefig(scatter2_path, dpi=300)
        plt.close()
        print(f"已輸出 Speed vs RMS 圖: {scatter2_path}")


if __name__ == "__main__":
    root = "."
    df_summary = collect_all_trials(root)
    if df_summary is not None:
        plot_statistics(df_summary)
