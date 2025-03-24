"""
Script to analyze PyTorch profiler results and display key metrics
"""

import os
import json
import pandas as pd
import argparse
from torch.profiler import profile
import torch

def load_and_analyze_profile(profile_path, specific_file=None):
    """Load and analyze the profiling data"""
    print("=== PyTorch Profiler Analysis ===\n")
    
    # Load the profiling data
    events = []
    if specific_file:
        # Handle single file
        if os.path.exists(specific_file):
            with open(specific_file, 'r') as f:
                trace_data = json.load(f)
                if 'traceEvents' in trace_data:
                    for event in trace_data['traceEvents']:
                        if 'cat' in event and 'dur' in event:
                            events.append({
                                'name': event.get('name', ''),
                                'category': event.get('cat', ''),
                                'duration_us': event.get('dur', 0),
                                'ts': event.get('ts', 0),
                                'args': event.get('args', {})
                            })
        else:
            print(f"Specified file {specific_file} does not exist!")
            return
    else:
        # Handle directory
        for root, dirs, files in os.walk(profile_path):
            for file in files:
                if file.endswith('.pt.trace.json'):
                    trace_path = os.path.join(root, file)
                    with open(trace_path, 'r') as f:
                        trace_data = json.load(f)
                        if 'traceEvents' in trace_data:
                            for event in trace_data['traceEvents']:
                                if 'cat' in event and 'dur' in event:
                                    events.append({
                                        'name': event.get('name', ''),
                                        'category': event.get('cat', ''),
                                        'duration_us': event.get('dur', 0),
                                        'ts': event.get('ts', 0),
                                        'args': event.get('args', {})
                                    })
    
    if not events:
        print("No profiling data found!")
        return
    
    # Convert to DataFrame
    events_df = pd.DataFrame(events)
    
    # Group by name and calculate statistics
    stats = events_df.groupby('name').agg({
        'duration_us': ['count', 'mean', 'sum']
    }).round(2)
    
    # Flatten column names
    stats.columns = ['count', 'avg_duration_us', 'total_duration_us']
    
    # Sort by total duration
    stats = stats.sort_values('total_duration_us', ascending=False)
    
    # Convert microseconds to milliseconds for better readability
    stats['avg_duration_ms'] = stats['avg_duration_us'] / 1000
    stats['total_duration_ms'] = stats['total_duration_us'] / 1000
    
    # Print summary
    print("Top 20 Most Time-Consuming Operations:")
    print("=====================================")
    summary = stats[['count', 'avg_duration_ms', 'total_duration_ms']].head(20)
    print(summary.to_string())
    print("\n")
    
    # Calculate total time spent in each major phase
    phases = ['forward', 'backward', 'optimizer_step', 'get_batch', 'train_step', 'evaluation']
    print("Time Spent in Major Phases:")
    print("==========================")
    for phase in phases:
        phase_data = stats.loc[stats.index.str.contains(phase, case=False)] if not stats.empty else None
        if phase_data is not None and not phase_data.empty:
            total_time = phase_data['total_duration_ms'].sum()
            avg_time = phase_data['avg_duration_ms'].mean()
            total_count = phase_data['count'].sum()
            print(f"{phase:15s}: {total_time:10.2f} ms "
                  f"(avg: {avg_time:10.2f} ms, "
                  f"count: {total_count})")
    
    # Print operation categories if available
    if 'category' in events_df.columns:
        print("\nTime by Operation Category:")
        print("==========================")
        cat_stats = events_df.groupby('category').agg({
            'duration_us': ['count', 'mean', 'sum']
        }).round(2)
        cat_stats.columns = ['count', 'avg_duration_us', 'total_duration_us']
        cat_stats['total_duration_ms'] = cat_stats['total_duration_us'] / 1000
        cat_stats['avg_duration_ms'] = cat_stats['avg_duration_us'] / 1000
        print(cat_stats[['count', 'avg_duration_ms', 'total_duration_ms']].sort_values('total_duration_ms', ascending=False).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze PyTorch profiler output')
    parser.add_argument('--file', type=str, help='Specific trace file to analyze')
    args = parser.parse_args()

    if args.file:
        load_and_analyze_profile(None, specific_file=args.file)
    else:
        profile_path = "./log/pytorch_profiler"
        if not os.path.exists(profile_path):
            print(f"Profile directory {profile_path} does not exist!")
        else:
            load_and_analyze_profile(profile_path) 