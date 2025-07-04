import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_gpu_memory_usage(filename):
    # Load CSV
    df = pd.read_csv(filename, on_bad_lines='skip')

    # Clean column names
    df.columns = df.columns.str.strip()
    print("Detected columns:", df.columns.tolist())

    # Filter valid rows
    df = df[df['timestamp'].str.contains(r'\d{4}/\d{2}/\d{2}', na=False)].copy()

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # Relative time in seconds
    start_time = df['timestamp'].min()
    df['relative_time_s'] = (df['timestamp'] - start_time).dt.total_seconds()

    # Convert memory usage from MiB to GB
    df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace('MiB', '', regex=False).str.strip()
    df['memory_used_gb'] = pd.to_numeric(df['memory.used [MiB]'], errors='coerce') / 1024
    df.dropna(subset=['memory_used_gb', 'index'], inplace=True)
    df['index'] = df['index'].astype(int)

    # Plot memory usage
    plt.figure(figsize=(10, 6))
    for gpu_index in sorted(df['index'].unique()):
        df_gpu = df[df['index'] == gpu_index]
        plt.plot(df_gpu['relative_time_s'], df_gpu['memory_used_gb'], label=f'GPU {gpu_index}')

    plt.xlabel("Time (s)")
    plt.ylabel("Memory Used (GB)")
    plt.title("GPU Memory Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save as PDF
    plt.savefig(pathname + "/gpu_memory_usage.pdf", format="pdf")
    # plt.show()  # Optional

if __name__ == "__main__":
    pathname = sys.argv[1] 
    # filename = sys.argv[1] if len(sys.argv) > 1 else "gpu_log.csv"
    plot_gpu_memory_usage(pathname + "/gpu_log.csv")
