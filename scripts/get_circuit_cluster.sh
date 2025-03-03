# Default values
DATABASE_PATH="clusters"
OUTPUT_DIR="artifacts"
NODE_THRESHOLD=1
EDGE_THRESHOLD=0.1
BATCH_SIZE=2
RUN="parameter-gradient-projections"
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
