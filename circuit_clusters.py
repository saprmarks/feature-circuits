#!/usr/bin/env python3
"""
Script to run circuit discovery on converted cluster data.
"""

import os
import argparse
import subprocess
import glob

def run_circuit_discovery(data_file, node_threshold, edge_threshold, output_dir, batch_size=2):
    """Run circuit discovery on a single data file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base name of the data file for naming the output
    base_name = os.path.basename(data_file)
    
    # Construct the command
    cmd = [
        "python", "circuit.py",
        "--model", "EleutherAI/pythia-70m-deduped",
        "--num_examples", "100",
        "--batch_size", str(batch_size),
        "--dataset", data_file,
        "--node_threshold", str(node_threshold),
        "--edge_threshold", str(edge_threshold),
        "--aggregation", "sum",
        "--nopair",
        "--circuit_dir", output_dir,
        "--plot_dir", os.path.join(output_dir, "figures")
    ]
    
    # Run the command
    print(f"Running circuit discovery on {data_file} with batch_size={batch_size}...")
    subprocess.run(cmd)
    print(f"Finished circuit discovery on {data_file}")

def run_on_all_clusters_in_run(run_name, clusters_dir, node_threshold, edge_threshold, output_dir, batch_size=2):
    """Run circuit discovery on all clusters in a run"""
    # Find all cluster files for this run
    cluster_files = glob.glob(os.path.join(clusters_dir, f"{run_name}_cluster_*.json"))
    
    if not cluster_files:
        print(f"No cluster files found for run {run_name}")
        return
    
    # Create a specific output directory for this run
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Run circuit discovery on each cluster
    for cluster_file in cluster_files:
        run_circuit_discovery(cluster_file, node_threshold, edge_threshold, run_output_dir, batch_size)

def run_on_all_runs(clusters_dir, node_threshold, edge_threshold, output_dir, batch_size=2):
    """Run circuit discovery on all clusters from all runs"""
    # Get all unique run names
    all_files = glob.glob(os.path.join(clusters_dir, "*_cluster_*.json"))
    run_names = set()
    
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        run_name = base_name.split("_cluster_")[0]
        run_names.add(run_name)
    
    # Run circuit discovery on each run
    for run_name in run_names:
        run_on_all_clusters_in_run(run_name, clusters_dir, node_threshold, edge_threshold, output_dir, batch_size)

def main():
    parser = argparse.ArgumentParser(description='Run circuit discovery on converted cluster data')
    parser.add_argument('--clusters-dir', type=str, default='data/clusters',
                        help='Directory containing the converted cluster JSON files')
    parser.add_argument('--output-dir', type=str, default='circuits/clusters',
                        help='Directory to save the circuit discovery results')
    parser.add_argument('--run', type=str, default=None,
                        help='Specific run to process (if not specified, process all runs)')
    parser.add_argument('--cluster', type=int, default=None,
                        help='Specific cluster to process (requires --run to be specified)')
    parser.add_argument('--node-threshold', type=float, default=0.2,
                        help='Node threshold for circuit discovery')
    parser.add_argument('--edge-threshold', type=float, default=0.02,
                        help='Edge threshold for circuit discovery')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for processing examples')
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.run and args.cluster is not None:
        # Run circuit discovery on a specific cluster
        cluster_file = os.path.join(args.clusters_dir, f"{args.run}_cluster_{args.cluster}")
        if os.path.exists(f"data/{cluster_file}.json"):
            cluster_output_dir = os.path.join(args.output_dir, args.run)
            run_circuit_discovery(cluster_file, args.node_threshold, args.edge_threshold, cluster_output_dir, args.batch_size)
        else:
            print(f"Cluster file {cluster_file} not found")
    elif args.run:
        # Run circuit discovery on all clusters in a specific run
        run_on_all_clusters_in_run(args.run, args.clusters_dir, args.node_threshold, args.edge_threshold, args.output_dir, args.batch_size)
    else:
        # Run circuit discovery on all clusters from all runs
        run_on_all_runs(args.clusters_dir, args.node_threshold, args.edge_threshold, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main() 