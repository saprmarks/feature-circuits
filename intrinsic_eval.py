import torch as t
import sys

def compute_triangle_effect(circuit, threshold):
    nodes = circuit["nodes"]
    total_effect = 0.
    total_triangle_effects = 0.
    square_effects = []
    triangle_effects = []

    for nodename in nodes:
        if nodename == "y":
            continue
        node = nodes[nodename]
        triangle_effect = node.resc.abs().item()
        triangle_effects.append(triangle_effect)
        total_triangle_effects += triangle_effect
        total_effect += triangle_effect

        zeros = t.zeros_like(node.act)
        circuit_feat_effects = t.where(node.act.abs() > threshold, node.act, zeros)
        square_effects.append(circuit_feat_effects.abs().sum().item())
        total_effect += node.act.abs().sum().item()
    
    square_effects = t.Tensor(square_effects)
    triangle_effects = t.Tensor(triangle_effects)
    square_sigmoid_weights = t.softmax(square_effects, dim=-1)
    triangle_sigmoid_weights = t.softmax(triangle_effects, dim=-1)
    weighted_squares = t.sum(square_sigmoid_weights * square_effects).item()
    weighted_triangles = t.sum(triangle_sigmoid_weights * triangle_effects).item()

    return weighted_squares / (weighted_squares + weighted_triangles)
    # return total_triangle_effects / total_effect

if __name__ == "__main__":
    circuit_path = sys.argv[1]
    node_threshold = float(circuit_path.split("_node")[1].split("_")[0])

    with open(circuit_path, "rb") as circuit_data:
        circuit = t.load(circuit_data)
    
    metric = compute_triangle_effect(circuit, node_threshold)
    print(metric)