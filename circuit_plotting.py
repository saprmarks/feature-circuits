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
            seq, feat = name.split(", ")
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

        # attn_i -> mlp_i
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


def plot_circuit(*args, **kwargs):
    raise NotImplementedError(
        "This function has been deprecated. Use plot_circuit_posaligned instead."
    )
