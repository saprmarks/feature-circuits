# Default values
DATABASE_PATH="data/clusters"
OUTPUT_DIR="circuits"
NODE_THRESHOLD=0.1
EDGE_THRESHOLD=0.01
BATCH_SIZE=1
RUN="parameter-gradient-projections"
DEVICE="cuda:0"
MAX_SEQUENCE_LENGTH=10
CLUSTER=1

# Run circuit discovery
python circuit_clusters.py \
    --clusters-dir $DATABASE_PATH \
    --output-dir $OUTPUT_DIR \
    --run $RUN \
    --cluster $CLUSTER \
    --node-threshold $NODE_THRESHOLD \
    --edge-threshold $EDGE_THRESHOLD \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --max-sequence-length $MAX_SEQUENCE_LENGTH
