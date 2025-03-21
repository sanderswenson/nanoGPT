"""
Script to analyze PyTorch profiler results and display key metrics
"""

import os
import pandas as pd
from torch.profiler import profile
import torch

def load_and_analyze_profile(profile_path):
    """Load and analyze the profiling data"""
    print("=== PyTorch Profiler Analysis ===\n")
    
    # Load the profiling data
    events_df = pd.DataFrame()
    for root, dirs, files in os.walk(profile_path):
        for file in files:
            if file.endswith('.pt.trace.json'):
                trace_path = os.path.join(root, file)
                df = pd.read_json(trace_path)
                events_df = pd.concat([events_df, df])
    
    if events_df.empty:
        print("No profiling data found!")
        return
    
    # Group by name and calculate statistics
    stats = events_df.groupby('name').agg({
        'dur': ['count', 'mean', 'sum'],
        'cpu_memory_usage': ['mean', 'max'],
        'cuda_memory_usage': ['mean', 'max']
    }).round(2)
    
    # Rename columns
    stats.columns = ['count', 'avg_duration_us', 'total_duration_us', 
                    'avg_cpu_mem_bytes', 'max_cpu_mem_bytes',
                    'avg_cuda_mem_bytes', 'max_cuda_mem_bytes']
    
    # Sort by total duration
    stats = stats.sort_values('total_duration_us', ascending=False)
    
    # Convert microseconds to milliseconds for better readability
    stats['avg_duration_ms'] = stats['avg_duration_us'] / 1000
    stats['total_duration_ms'] = stats['total_duration_us'] / 1000
    
    # Convert bytes to MB for better readability
    for col in ['avg_cpu_mem_bytes', 'max_cpu_mem_bytes', 
                'avg_cuda_mem_bytes', 'max_cuda_mem_bytes']:
        stats[col.replace('bytes', 'MB')] = stats[col] / (1024 * 1024)
    
    # Print summary
    print("Top 10 Most Time-Consuming Operations:")
    print("=====================================")
    summary = stats[['count', 'avg_duration_ms', 'total_duration_ms']].head(10)
    print(summary.to_string())
    print("\n")
    
    print("Memory Usage (Top 10 by CUDA memory):")
    print("====================================")
    memory = stats[['avg_cpu_mem_MB', 'max_cpu_mem_MB', 
                   'avg_cuda_mem_MB', 'max_cuda_mem_MB']].head(10)
    print(memory.to_string())
    print("\n")
    
    # Calculate total time spent in each major phase
    phases = ['forward', 'backward', 'optimizer_step', 'get_batch']
    print("Time Spent in Major Phases:")
    print("==========================")
    for phase in phases:
        phase_data = stats.loc[phase] if phase in stats.index else None
        if phase_data is not None:
            print(f"{phase:15s}: {phase_data['total_duration_ms']:10.2f} ms "
                  f"(avg: {phase_data['avg_duration_ms']:10.2f} ms, "
                  f"count: {phase_data['count']})")

if __name__ == "__main__":
    profile_path = "./log/pytorch_profiler"
    if not os.path.exists(profile_path):
        print(f"Profile directory {profile_path} does not exist!")
    else:
        load_and_analyze_profile(profile_path) 