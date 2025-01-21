import os
from collections import defaultdict
from graphviz import Digraph
import math

# Constants
EXAMPLE_FONT = 36  # Font size for example words at the bottom
CHAR_WIDTH = 0.1  # Width per character in labels
NODE_BUFFER = 0.1  # Extra width added to each node
INTERNODE_BUFFER = 0.1  # Extra space between nodes (not added to node width)
MIN_COL_WIDTH = 1  # Minimum width for a column
VERTICAL_SPACING = 0.1  # Vertical spacing between nodes in the same column
NODE_HEIGHT = 0.4  # Height of each node
CIRCUIT_ROW_LABEL_X = -10  # X position for row labels in plot_circuit


def get_name_pythia(component, layer, idx):
    match idx:
        case (seq, feat):
            if feat == 32768:
                feat = "ε"
            if layer == -1:
                return f"{seq}, embed/{feat}"
            return f"{seq}, {component}_{layer}/{feat}"
        case (feat,):
            if feat == 32768:
                feat = "ε"
            if layer == -1:
                return f"embed/{feat}"
            return f"{component}_{layer}/{feat}"
        case _:
            raise ValueError(f"Invalid idx: {idx}")


def get_name_gemma(component, layer, idx):
    match idx:
        case (seq, feat):
            if feat == 2**14:
                feat = "ε"
            return f"{seq}, {component}_{layer}/{feat}"
        case (feat,):
            if feat == 2**14:
                feat = "ε"
            return f"{component}_{layer}/{feat}"
        case _:
            raise ValueError(f"Invalid idx: {idx}")


def plot_circuit_posaligned(
    nodes,
    edges,
    layers=6,
    example_text="The managers that the parent likes",
    node_threshold=0.1,
    edge_threshold=0.01,
    pen_thickness=3,
    horizontal_spacing=0.2,
    annotations=None,
    save_dir="circuit",
    gemma_mode=False,
    parallel_attn=True,
):
    get_name = get_name_gemma if gemma_mode else get_name_pythia

    def to_hex(number):
        scale = max(
            abs(min([v.to_tensor().min() for n, v in nodes.items() if n != "y"])),
            abs(max([v.to_tensor().max() for n, v in nodes.items() if n != "y"])),
        )
        number = number / scale

        if number < 0:
            red = 255
            green = blue = int((1 + number) * 255)
        elif number > 0:
            blue = 255
            red = green = int((1 - number) * 255)
        else:
            red = green = blue = 255

        text_hex = (
            "#000000"
            if (red * 0.299 + green * 0.587 + blue * 0.114) > 170
            else "#ffffff"
        )
        hex_code = f"#{red:02X}{green:02X}{blue:02X}"

        return hex_code, text_hex

    def split_label(label):
        if len(label) > 20:  # Add a line break for labels longer than 20 characters
            if "/" in label:
                split_index = (
                    label.find("/", 10) + 1
                )  # Find the first '/' after the 10th character
                if split_index > 0:
                    return label[:split_index], label[split_index:]
            words = label.split()
            mid = math.ceil(len(words) / 2)
            return " ".join(words[:mid]), " ".join(words[mid:])
        return label, ""

    if annotations is None:
        def get_label(name):
            return split_label(name.split(", ")[-1])  # Remove sequence information
    else:
        def get_label(name):
            _, feat = name.split(", ")
            if feat in annotations:
                return split_label(annotations[feat])
            return split_label(feat)  # Remove sequence information

    G = Digraph(name="Feature circuit")
    G.graph_attr.update(rankdir="BT", newrank="true")
    G.node_attr.update(shape="box", style="rounded")

    words = example_text.split()
    if gemma_mode:
        words = ["<BOS>"] + words
    seq_len = nodes[list(nodes.keys())[0]].act.shape[0]
    assert (length := len(words)) == seq_len, (
        "The number of words in example_text should match the sequence length"
    )

    nodes_by_submod = {}
    if not gemma_mode:
        nodes_by_submod["embed"] = {
            tuple(idx.tolist()): nodes["embed"].to_tensor()[tuple(idx)].item()
            for idx in (nodes["embed"].to_tensor().abs() > node_threshold).nonzero()
        }
    nodes_by_seqpos = defaultdict(lambda: defaultdict(list))

    for layer in range(layers):
        for component in ["attn", "mlp", "resid"]:
            submod_nodes = nodes[f"{component}_{layer}"].to_tensor()
            nodes_by_submod[f"{component}_{layer}"] = {
                tuple(idx.tolist()): submod_nodes[tuple(idx)].item()
                for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }

    # Calculate node widths and column widths
    node_widths = {}
    column_widths = [0] * length

    for submod, submod_nodes in nodes_by_submod.items():
        component = submod.split("_")[0]
        layer = -1 if component == "embed" else int(submod.split("_")[1])
        for idx, _ in submod_nodes.items():
            name = get_name(component, layer, idx)
            label_line1, label_line2 = get_label(name)
            width = (
                max(CHAR_WIDTH * len(label_line1), CHAR_WIDTH * len(label_line2))
                + NODE_BUFFER
            )
            node_widths[name] = width

            seq = int(name.split(",")[0]) if "," in name else -1  # -1 for global
            if seq != -1:
                nodes_by_seqpos[seq][submod].append(name)
                column_widths[seq] = max(
                    column_widths[seq],
                    sum(node_widths[n] for n in nodes_by_seqpos[seq][submod])
                    + (len(nodes_by_seqpos[seq][submod]) - 1) * INTERNODE_BUFFER,
                )

    # Ensure minimum column width
    column_widths = [max(width, MIN_COL_WIDTH) for width in column_widths]

    # Calculate positions
    node_positions = {}
    y_offset = 0
    components = (["embed"] if not gemma_mode else []) + [
        f"{comp}_{layer}"
        for layer in range(layers)
        for comp in ["attn", "mlp", "resid"]
    ]
    for submod in components:
        x_offset = 0
        for seq in range(length):
            cell_x_offset = x_offset + column_widths[seq] / 2  # Center of the column
            total_width = (
                sum(node_widths[n] for n in nodes_by_seqpos[seq][submod])
                + (len(nodes_by_seqpos[seq][submod]) - 1) * INTERNODE_BUFFER
            )
            start_x = cell_x_offset - total_width / 2
            for name in nodes_by_seqpos[seq][submod]:
                node_center = start_x + node_widths[name] / 2
                node_positions[name] = (node_center, y_offset)
                start_x += node_widths[name] + INTERNODE_BUFFER
            x_offset += column_widths[seq] + horizontal_spacing

        # Add row label
        G.node(
            f"row_{submod}", label=submod, pos=f"{-2},{y_offset}!", shape="plaintext"
        )

        y_offset += NODE_HEIGHT + VERTICAL_SPACING

    # Add nodes to the graph
    for submod, submod_nodes in nodes_by_submod.items():
        component = submod.split("_")[0]
        layer = -1 if component == "embed" else int(submod.split("_")[1])
        for idx, effect in submod_nodes.items():
            name = get_name(component, layer, idx)
            fillhex, texthex = to_hex(effect)
            x, y = node_positions[name]

            label_line1, label_line2 = get_label(name)
            label = f"{label_line1}\\n{label_line2}" if label_line2 else label_line1
            is_epsilon = "ε" in name
            node_shape = "triangle" if is_epsilon else "box"

            G.node(
                name,
                label=label,
                pos=f"{x},{y}!",
                width=str(node_widths[name]),
                height=str(NODE_HEIGHT),
                fixedsize="true",
                fillcolor=fillhex,
                fontcolor=texthex,
                style="filled",
                shape=node_shape,
            )

    # Add edges
    for layer in range(layers):
        prev_layer = layer - 1
        prev_component = (
            "embed" if layer == 0 and not gemma_mode else f"resid_{prev_layer}"
        )

        if layer == 0 and gemma_mode:
            # Skip connections from resid_{-1} in gemma_mode
            pass
        else:
            # resid_{i-1} -> attn_i
            for upstream_idx in nodes_by_submod[prev_component].keys():
                for downstream_idx in nodes_by_submod[f"attn_{layer}"].keys():
                    weight = edges[prev_component][f"attn_{layer}"][
                        tuple(downstream_idx)
                    ][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(
                            "embed" if layer == 0 and not gemma_mode else "resid",
                            prev_layer,
                            upstream_idx,
                        )
                        dname = get_name("attn", layer, downstream_idx)
                        G.edge(
                            uname,
                            dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color="red" if weight < 0 else "blue",
                        )

            # resid_{i-1} -> mlp_i
            for upstream_idx in nodes_by_submod[prev_component].keys():
                for downstream_idx in nodes_by_submod[f"mlp_{layer}"].keys():
                    weight = edges[prev_component][f"mlp_{layer}"][
                        tuple(downstream_idx)
                    ][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(
                            "embed" if layer == 0 and not gemma_mode else "resid",
                            prev_layer,
                            upstream_idx,
                        )
                        dname = get_name("mlp", layer, downstream_idx)
                        G.edge(
                            uname,
                            dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color="red" if weight < 0 else "blue",
                        )

            # resid_{i-1} -> resid_i
            for upstream_idx in nodes_by_submod[prev_component].keys():
                for downstream_idx in nodes_by_submod[f"resid_{layer}"].keys():
                    weight = edges[prev_component][f"resid_{layer}"][
                        tuple(downstream_idx)
                    ][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(
                            "embed" if layer == 0 and not gemma_mode else "resid",
                            prev_layer,
                            upstream_idx,
                        )
                        dname = get_name("resid", layer, downstream_idx)
                        G.edge(
                            uname,
                            dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color="red" if weight < 0 else "blue",
                        )

        # attn_i -> mlp_i (only if parallel_attn is False)
        if not parallel_attn:
            for upstream_idx in nodes_by_submod[f"attn_{layer}"].keys():
                for downstream_idx in nodes_by_submod[f"mlp_{layer}"].keys():
                    weight = edges[f"attn_{layer}"][f"mlp_{layer}"][tuple(downstream_idx)][
                        tuple(upstream_idx)
                    ].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name("attn", layer, upstream_idx)
                        dname = get_name("mlp", layer, downstream_idx)
                        G.edge(
                            uname,
                            dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color="red" if weight < 0 else "blue",
                        )

        # attn_i -> resid_i
        for upstream_idx in nodes_by_submod[f"attn_{layer}"].keys():
            for downstream_idx in nodes_by_submod[f"resid_{layer}"].keys():
                weight = edges[f"attn_{layer}"][f"resid_{layer}"][
                    tuple(downstream_idx)
                ][tuple(upstream_idx)].item()
                if abs(weight) > edge_threshold:
                    uname = get_name("attn", layer, upstream_idx)
                    dname = get_name("resid", layer, downstream_idx)
                    G.edge(
                        uname,
                        dname,
                        penwidth=str(abs(weight) * pen_thickness),
                        color="red" if weight < 0 else "blue",
                    )

        # mlp_i -> resid_i
        for upstream_idx in nodes_by_submod[f"mlp_{layer}"].keys():
            for downstream_idx in nodes_by_submod[f"resid_{layer}"].keys():
                weight = edges[f"mlp_{layer}"][f"resid_{layer}"][tuple(downstream_idx)][
                    tuple(upstream_idx)
                ].item()
                if abs(weight) > edge_threshold:
                    uname = get_name("mlp", layer, upstream_idx)
                    dname = get_name("resid", layer, downstream_idx)
                    G.edge(
                        uname,
                        dname,
                        penwidth=str(abs(weight) * pen_thickness),
                        color="red" if weight < 0 else "blue",
                    )

    # Add word labels at the bottom (centered)
    word_y_offset = -1
    x_offset = 0
    for i, word in enumerate(words):
        word_x = x_offset + column_widths[i] / 2
        G.node(
            f"word_{i}",
            label=word,
            pos=f"{word_x},{word_y_offset}!",
            shape="plaintext",
            fontsize=str(EXAMPLE_FONT),
        )
        x_offset += column_widths[i] + horizontal_spacing
    last_word_x = word_x

    # Add 'y' node
    total_width = sum(column_widths) + (length - 1) * horizontal_spacing
    G.node("y", shape="diamond", pos=f"{last_word_x},{y_offset}!")
    for idx in nodes_by_submod[f"resid_{layers - 1}"].keys():
        weight = edges[f"resid_{layers - 1}"]["y"][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name("resid", layers - 1, idx)
            G.edge(
                name,
                "y",
                penwidth=str(abs(weight) * pen_thickness),
                color="red" if weight < 0 else "blue",
            )

    # Add dashed vertical gray lines separating columns
    x_offset = 0
    for i in range(length + 1):
        line_top = y_offset + 1  # Top of the image
        line_bottom = word_y_offset - 1  # Just below the words at the bottom
        G.node(
            f"line_top_{i}",
            shape="point",
            pos=f"{x_offset},{line_top}!",
            width="0",
            height="0",
        )
        G.node(
            f"line_bottom_{i}",
            shape="point",
            pos=f"{x_offset},{line_bottom}!",
            width="0",
            height="0",
        )
        G.edge(
            f"line_top_{i}",
            f"line_bottom_{i}",
            style="dashed",
            color="gray",
            penwidth=str(0.5 * pen_thickness),
            dir="none",  # This removes the arrow direction
            arrowhead="none",
        )  # This explicitly removes arrowheads
        if i < length:
            x_offset += column_widths[i] + horizontal_spacing

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format="png", cleanup=True, engine="neato")


def plot_circuit(
    nodes,
    edges,
    layers=6,
    node_threshold=0.1,
    edge_threshold=0.01,
    pen_thickness=3,
    annotations=None,
    save_dir="circuit",
    gemma_mode=False,
    parallel_attn=True,
):
    """Plot a circuit where weights have been aggregated across sequence positions.
    
    Args:
        nodes: Dict mapping submodule names to SparseAct objects
        edges: Dict mapping upstream submodule names to dict mapping downstream submodule names 
              to tensors of shape [downstream_feats, upstream_feats]
        layers: Number of layers in the model
        node_threshold: Threshold for including nodes based on their effect size
        edge_threshold: Threshold for including edges based on their weight
        pen_thickness: Base thickness for drawing edges
        annotations: Optional dict mapping feature names to human-readable descriptions
        save_dir: Directory to save the plot
        gemma_mode: Whether to use Gemma-specific naming
        parallel_attn: Whether to draw edges between attention and MLP layers
    """
    get_name = get_name_gemma if gemma_mode else get_name_pythia

    def to_hex(number):
        scale = max(
            abs(min([v.to_tensor().min() for n, v in nodes.items() if n != "y"])),
            abs(max([v.to_tensor().max() for n, v in nodes.items() if n != "y"])),
        )
        number = number / scale

        if number < 0:
            red = 255
            green = blue = int((1 + number) * 255)
        elif number > 0:
            blue = 255
            red = green = int((1 - number) * 255)
        else:
            red = green = blue = 255

        text_hex = (
            "#000000"
            if (red * 0.299 + green * 0.587 + blue * 0.114) > 170
            else "#ffffff"
        )
        hex_code = f"#{red:02X}{green:02X}{blue:02X}"

        return hex_code, text_hex

    def split_label(label):
        if len(label) > 20:
            if "/" in label:
                split_index = label.find("/", 10) + 1
                if split_index > 0:
                    return label[:split_index], label[split_index:]
            words = label.split()
            mid = math.ceil(len(words) / 2)
            return " ".join(words[:mid]), " ".join(words[mid:])
        return label, ""

    if annotations is None:
        def get_label(name):
            return split_label(name)
    else:
        def get_label(name):
            if name in annotations:
                return split_label(annotations[name])
            return split_label(name)

    # Create graph
    G = Digraph(name="Feature circuit")
    G.graph_attr.update(rankdir="BT", newrank="true")  # newrank needed for proper alignment
    G.node_attr.update(shape="box", style="rounded")

    # Track nodes by sublayer for ranking
    nodes_by_sublayer = defaultdict(list)
    
    # Add nodes
    components = (["embed"] if not gemma_mode else []) + [
        f"{comp}_{layer}"
        for layer in range(layers)
        for comp in ["attn", "mlp", "resid"]
    ]

    # First pass: collect nodes and calculate widths
    node_info = {}  # Store node info (label, color, etc.) for later use
    node_widths = {}  # Store width of each node

    for submod, values in nodes.items():
        if submod == "y":
            continue
        
        # Determine layer number and component
        if submod == "embed":
            layer = -1
            component = "embed"
        else:
            component, layer = submod.split("_")
            layer = int(layer)
            
        # Add nodes that exceed threshold
        values = values.to_tensor()
        for feat_idx in range(values.shape[0]):
            if abs(values[feat_idx]) > node_threshold:
                name = get_name(component, layer, (feat_idx,))
                fillhex, texthex = to_hex(values[feat_idx])
                label_line1, label_line2 = get_label(name)
                label = f"{label_line1}\\n{label_line2}" if label_line2 else label_line1
                
                is_epsilon = feat_idx == values.shape[0] - 1
                node_shape = "triangle" if is_epsilon else "box"
                
                # Store node info
                node_info[name] = {
                    "label": label,
                    "fillcolor": fillhex,
                    "fontcolor": texthex,
                    "shape": node_shape,
                }
                
                # Calculate node width
                width = max(CHAR_WIDTH * len(label_line1), CHAR_WIDTH * len(label_line2)) + NODE_BUFFER
                node_widths[name] = width
                
                sublayer = f"{component}_{layer}" if layer >= 0 else "embed"
                nodes_by_sublayer[sublayer].append(name)

    # Second pass: calculate positions and add nodes
    y_offset = 0
    for i, submod in enumerate(components):
        # Add row label
        G.node(
            f"row_{submod}",
            label=submod,
            pos=f"{CIRCUIT_ROW_LABEL_X},{y_offset}!",  # Use the new constant
            shape="plaintext"
        )
        
        # Position nodes in this sublayer
        nodes_in_layer = nodes_by_sublayer[submod]
        if nodes_in_layer:
            total_width = sum(node_widths[n] for n in nodes_in_layer) + INTERNODE_BUFFER * (len(nodes_in_layer) - 1)
            total_width = max(total_width, MIN_COL_WIDTH)  # Ensure minimum width
            start_x = -total_width / 2  # Center the nodes
            
            for node_name in nodes_in_layer:
                node_info_dict = node_info[node_name]
                x_pos = start_x + node_widths[node_name] / 2
                
                G.node(
                    node_name,
                    label=node_info_dict["label"],
                    fillcolor=node_info_dict["fillcolor"],
                    fontcolor=node_info_dict["fontcolor"],
                    style="filled",
                    shape=node_info_dict["shape"],
                    pos=f"{x_pos},{y_offset}!",
                    width=str(node_widths[node_name]),
                    height=str(NODE_HEIGHT),
                    fixedsize="true",
                )
                
                start_x += node_widths[node_name] + INTERNODE_BUFFER
        
        y_offset += NODE_HEIGHT + VERTICAL_SPACING  # Proper vertical spacing between layers

    # Add edges
    for upstream_mod, downstream_dict in edges.items():
        for downstream_mod, weight_matrix in downstream_dict.items():
            # Skip attention->mlp edges if using parallel attention
            if parallel_attn and upstream_mod.startswith("attn_") and downstream_mod.startswith("mlp_"):
                continue
                
            # Get component and layer info for upstream and downstream
            if upstream_mod == "embed":
                u_component = "embed"
                u_layer = -1
            else:
                u_component, u_layer = upstream_mod.split("_")
                u_layer = int(u_layer)
                
            if downstream_mod == "embed":
                d_component = "embed"
                d_layer = -1
            elif downstream_mod == "y":
                d_component = "y"
                d_layer = layers
            else:
                d_component, d_layer = downstream_mod.split("_")
                d_layer = int(d_layer)

            if downstream_mod == "y":
                # For y node, weight_matrix is just a vector [n_upstream_feats]
                upstream_values = nodes[upstream_mod].to_tensor()
                for idx in range(weight_matrix.shape[0]):
                    if abs(upstream_values[idx]) > node_threshold:
                        name = get_name(u_component, u_layer, (idx,))
                        weight = weight_matrix[idx].item()
                        if abs(weight) > edge_threshold:
                            G.edge(
                                name,
                                "y",
                                penwidth=str(abs(weight) * pen_thickness),
                                color="red" if weight < 0 else "blue",
                            )
            else:
                # For normal edges, weight_matrix is [downstream_feats, upstream_feats]
                upstream_values = nodes[upstream_mod].to_tensor()
                downstream_values = nodes[downstream_mod].to_tensor()
                
                for d_idx in range(weight_matrix.shape[0]):
                    if abs(downstream_values[d_idx]) > node_threshold:
                        d_name = get_name(d_component, d_layer, (d_idx,))
                        for u_idx in range(weight_matrix.shape[1]):
                            if abs(upstream_values[u_idx]) > node_threshold:
                                u_name = get_name(u_component, u_layer, (u_idx,))
                                weight = weight_matrix[d_idx, u_idx].item()
                                if abs(weight) > edge_threshold:
                                    G.edge(
                                        u_name,
                                        d_name,
                                        penwidth=str(abs(weight) * pen_thickness),
                                        color="red" if weight < 0 else "blue",
                                    )

    # Add y node at top with position above the last row
    G.node("y", shape="diamond", pos=f"0,{y_offset}!")
    
    # Create directory if needed
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
        
    # Render graph
    G.render(save_dir, format="png", cleanup=True, engine="neato")
