import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import sys

# Parse train_small.log
def plot_gpu_memory_info(filename):
    iteration_data = []
    log_file = filename


    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+iteration\s+(\d+)/\s*(\d+)", line)
            if match:
                timestamp_str, iter_num, total_iters = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                iteration_data.append((timestamp, int(iter_num)))

    # Convert to DataFrame
    df_iters = pd.DataFrame(iteration_data, columns=["timestamp", "iteration"])

    # Load GPU memory usage data
    df_gpu = pd.read_csv(pathname + "/gpu_log.csv", on_bad_lines='skip')
    df_gpu.columns = df_gpu.columns.str.strip()
    df_gpu = df_gpu[df_gpu['timestamp'].str.contains(r'\d{4}/\d{2}/\d{2}', na=False)].copy()
    df_gpu['timestamp'] = pd.to_datetime(df_gpu['timestamp'], errors='coerce')
    df_gpu.dropna(subset=['timestamp'], inplace=True)
    df_gpu['relative_time_s'] = (df_gpu['timestamp'] - df_gpu['timestamp'].min()).dt.total_seconds()
    df_gpu['memory_used_gb'] = pd.to_numeric(df_gpu['memory.used [MiB]'].str.replace('MiB', '', regex=False).str.strip(), errors='coerce') / 1024
    df_gpu.dropna(subset=['memory_used_gb', 'index'], inplace=True)
    df_gpu['index'] = df_gpu['index'].astype(int)

    # Compute relative time for iterations
    df_iters['relative_time_s'] = (df_iters['timestamp'] - df_gpu['timestamp'].min()).dt.total_seconds()

    # Plot
    plt.figure(figsize=(10, 6))
    for gpu_index in sorted(df_gpu['index'].unique()):
        df_gpu_i = df_gpu[df_gpu['index'] == gpu_index]
        plt.plot(df_gpu_i['relative_time_s'], df_gpu_i['memory_used_gb'], label=f'GPU {gpu_index}')

    # Plot iteration markers
        # Filter iteration markers to every N-th (e.g., every 50 iterations)
    iter_step = 25
    df_iters_filtered = df_iters[df_iters['iteration'] % iter_step == 0]
    for _, row in df_iters_filtered.iterrows():
        plt.axvline(row['relative_time_s'], color='gray', linestyle='--', linewidth=0.7)
        plt.text(row['relative_time_s'], 1, f"Iter {row['iteration']}", rotation=90, fontsize=8, va='bottom')

    # for _, row in df_iters.iterrows():
    #     plt.axvline(row['relative_time_s'], color='gray', linestyle='--', linewidth=0.7)
    #     plt.text(row['relative_time_s'], 1, f"Iter {row['iteration']}", rotation=90, fontsize=8, va='bottom')

    plt.xlabel("Time (s)")
    plt.ylabel("Memory Used (GB)")
    plt.title("GPU Memory Usage Over Time with Iteration Markers")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pathname + "/gpu_memory_with_iterations.pdf", format="pdf")


if __name__ == "__main__":
    pathname = sys.argv[1] 
    # log_file = 'train_small.log'
    # filename = sys.argv[1] if len(sys.argv) > 1 else "gpu_log.csv"
    plot_gpu_memory_info(pathname + "/train_small.log")