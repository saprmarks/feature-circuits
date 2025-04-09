#!/bin/bash

# Common parameters
DATABASE_PATH="data/clusters"
OUTPUT_DIR="circuits"
NODE_THRESHOLD=0.1
EDGE_THRESHOLD=0.01
BATCH_SIZE=1
MAX_SEQUENCE_LENGTH=64

# Create logs directory if it doesn't exist
mkdir -p logs
# Store parent PID
echo $$ > logs/circuit_clusters_parent.pid
echo "" > logs/circuit_clusters_children.pids

# Array of configurations
declare -A CONFIGS=(
    [1]="sae-features_lin-effects_sum-over-pos_nsamples8192_nctx64"
    [2]="sae-features_lin-effects_final-5-pos_nsamples8192_nctx64"
    [3]="sae-features_lin-effects_final-1-pos_nsamples8192_nctx64"
    [4]="sae-features_activations_sum-over-pos_nsamples8192_nctx64"
    [5]="sae-features_activations_final-5-pos_nsamples8192_nctx64"
    [6]="sae-features_activations_final-1-pos_nsamples8192_nctx64"
    [7]="parameter-gradient-projections"
)



# Function to run a single configuration
run_config() {
    local gpu_id=$1
    local run_name=$2
    
    local device="cuda:$gpu_id"
    local log_filename="circuit_discovery_${run_name}_node${NODE_THRESHOLD}_edge${EDGE_THRESHOLD}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting run $run_name on $device..."
    
    nohup python circuit_clusters.py \
        --clusters-dir $DATABASE_PATH \
        --output-dir $OUTPUT_DIR \
        --run $run_name \
        --node-threshold $NODE_THRESHOLD \
        --edge-threshold $EDGE_THRESHOLD \
        --batch-size $BATCH_SIZE \
        --device $device \
        --max-sequence-length $MAX_SEQUENCE_LENGTH \
        > "logs/${log_filename}" 2>&1 &
    
    local pid=$!
    echo $pid >> logs/circuit_clusters_children.pids
    echo "Started circuit discovery for $run_name in background with PID $pid"
    echo "Log file: logs/${log_filename}"
    echo "Monitor with: tail -f logs/${log_filename}"
    echo "----------------------------------------"
}

# Run all configurations
for gpu_id in "${!CONFIGS[@]}"; do
    run_name="${CONFIGS[$gpu_id]}"

    run_config $gpu_id "$run_name"
    # Small delay between starting jobs
    sleep 2
done

echo "All circuit discovery jobs have been started."
echo "Use 'ps aux | grep circuit_clusters.py' to check running processes"
echo "Use 'nvidia-smi' to monitor GPU usage"
echo "Parent PID stored in logs/circuit_clusters_parent.pid"
echo "Child PIDs stored in logs/circuit_clusters_children.pids"
echo "To kill all processes, run: kill \$(cat logs/circuit_clusters_children.pids) \$(cat logs/circuit_clusters_parent.pid)" 