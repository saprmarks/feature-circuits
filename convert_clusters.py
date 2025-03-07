#!/usr/bin/env python3
"""
Script to convert cluster data from data-cluster directory to JSON files
that can be read by load_examples_nopair function.
"""

import pickle
import gzip
import io
import json
import os
from sqlitedict import SqliteDict
from huggingface_hub import hf_hub_download
import tarfile
# Update this path to match your environment
DEFAULT_DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def download_clusters_from_hf(local_dir):
    repo_id = "canrager/sfc_data"
    fname = "data.tar.gz"
    hf_hub_download(
        repo_id,
        filename=fname,
        local_dir=local_dir,
        repo_type="dataset",
    )
    # unzip the downloaded file into a new subdirectory called clusters
    local_path = os.path.join(local_dir, fname)
    path_to_extract = os.path.join(local_dir, "clusters_raw")
    os.makedirs(path_to_extract, exist_ok=True)
    with tarfile.open(local_path, 'r:gz') as tar:
        tar.extractall(path=path_to_extract)
    # remove the tar file
    # os.remove(local_path)
    return path_to_extract

def list_databases(database_path):
    """List all database directories in the data folder"""
    databases = []
    for item in os.listdir(database_path):
        if os.path.isdir(os.path.join(database_path, item)) and os.path.exists(f"{database_path}/{item}/database.sqlite"):
            databases.append(item)
    return databases

def get_cluster_count(database_path, db_name):
    """Get number of clusters in a database"""
    with SqliteDict(f"{database_path}/{db_name}/database.sqlite") as db:
        return len(db)

def load_cluster_data(database_path, db_name, cluster_idx):
    """Load data for a specific cluster"""
    with SqliteDict(f"{database_path}/{db_name}/database.sqlite") as db:
        compressed_bytes = db[cluster_idx]
        decompressed_object = io.BytesIO(compressed_bytes)
        with gzip.GzipFile(fileobj=decompressed_object, mode='rb') as file:
            cluster_data = pickle.load(file)
    return cluster_data

def convert_cluster_to_json(database_path, db_name, cluster_idx, output_dir):
    """Convert a cluster to a JSON file that can be read by load_examples_nopair"""
    cluster_data = load_cluster_data(database_path, db_name, cluster_idx)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the data for load_examples_nopair
    formatted_data = {}
    for i, (key, context) in enumerate(cluster_data['contexts'].items()):
        # Join the context tokens into a string
        context_text = "".join(context['context'])
        answer_text = context['answer']
        
        formatted_data[str(i)] = {
            "context": context_text,
            "answer": answer_text
        }
    
    # Save to a JSON file
    output_file = os.path.join(output_dir, f"{db_name}_cluster_{cluster_idx}.json")
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    return output_file

def convert_all_clusters_in_run(database_path, db_name, output_dir):
    """Convert all clusters in a run to JSON files"""
    n_clusters = get_cluster_count(database_path, db_name)
    output_files = []
    
    for cluster_idx in range(n_clusters):
        try:
            output_file = convert_cluster_to_json(database_path, db_name, cluster_idx, output_dir)
            output_files.append(output_file)
            print(f"Converted cluster {cluster_idx} from {db_name} to {output_file}")
        except Exception as e:
            print(f"Error converting cluster {cluster_idx} from {db_name}: {e}")
    
    return output_files

def convert_all_runs(database_path, output_dir):
    """Convert all clusters from all runs to JSON files"""
    databases = list_databases(database_path)
    print(f"Found {len(databases)} databases")
    print(f'database path: {database_path}')
    all_output_files = []
    
    for db_name in databases:
        output_files = convert_all_clusters_in_run(database_path, db_name, output_dir)
        all_output_files.extend(output_files)
    
    return all_output_files

def main():
    database_path = DEFAULT_DATABASE_PATH
    # download the clusters from the huggingface hub
    download_clusters_from_hf(database_path)
    output_dir = os.path.join(database_path, "clusters")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    database_path = os.path.join(database_path, "clusters_raw", "data")
    output_files = convert_all_runs(database_path, output_dir)
    print(f"Converted {len(output_files)} clusters from all runs")


if __name__ == "__main__":
    main() 