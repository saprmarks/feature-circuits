from graphviz import Digraph
from collections import defaultdict
import re
import os

# def get_name(component, layer, idx):
#     match idx:
#         case (seq, feat):
#             if feat == 32768: feat = 'res'
#             return f'{seq}, {component}_{layer}/{feat}'
#         case (feat,):
#             if feat == 32768: feat = 'res'
#             return f'{component}_{layer}/{feat}'
#         case _: raise ValueError(f"Invalid idx: {idx}")

def get_name(component, layer, idx):
    if component == 'resid': layer += 1
    match idx:
        case (seq, feat):
            if feat == 32768: feat = 'err'
            return f'{seq}, {component}_{layer}/{feat}'
        case (feat,):
            if feat == 32768: feat = 'err'
            return f'{component}_{layer}/{feat}'
        case _: raise ValueError(f"Invalid idx: {idx}")


def plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, pen_thickness=1, annotations=None, save_dir='circuit'):

    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))

    # for deciding shade of node
    def to_hex(number):
        number = number / scale
        
        # Define how the intensity changes based on the number
        # - Negative numbers increase red component to max
        # - Positive numbers increase blue component to max
        # - 0 results in white
        if number < 0:
            # Increase towards red, full intensity at -1.0
            red = 255
            green = blue = int((1 + number) * 255)  # Increase other components less as it gets more negative
        elif number > 0:
            # Increase towards blue, full intensity at 1.0
            blue = 255
            red = green = int((1 - number) * 255)  # Increase other components less as it gets more positive
        else:
            # Exact 0, resulting in white
            red = green = blue = 255 
        
        # decide whether text is black or white depending on darkness of color
        text_hex = "#000000" if (red*0.299 + green*0.587 + blue*0.114) > 170 else "#ffffff"

        # Convert to hex, ensuring each component is 2 digits
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        
        return hex_code, text_hex
    
    if annotations is None:
        def get_label(name):
            return name
    else:
        def get_label(name):
            match name.split(', '):
                case seq, feat:
                    if feat in annotations:
                        component = feat.split('/')[0]
                        component = feat.split('_')[0]
                        return f'{seq}, {annotations[feat]} ({component})'
                    return name
                case [feat]:
                    if feat in annotations:
                        component = feat.split('/')[0]
                        component = feat.split('_')[0]
                        return f'{annotations[feat]} ({component})'

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    nodes_by_submod = {
        'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
    }
    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }
    edges['resid_-1'] = edges['embed']

    for layer in range(-1, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1 and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    fillhex, texthex = to_hex(effect)
                    if name[-3:] == 'err':
                        subgraph.node(name, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
        
        # add edges to previous layer resid
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
                for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                    weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name('resid', layer-1, upstream_idx)
                        dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=True)


def plot_circuit_posaligned(nodes, edges, layers=6, length=6, example_text="The managers that the parent likes",
                            node_threshold=0.1, edge_threshold=0.01, pen_thickness=3, annotations=None, save_dir='circuit'):

    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))

    words = example_text.split()

    # for deciding shade of node
    def to_hex(number):
        number = number / scale
        
        # Define how the intensity changes based on the number
        # - Negative numbers increase red component to max
        # - Positive numbers increase blue component to max
        # - 0 results in white
        if number < 0:
            # Increase towards red, full intensity at -1.0
            red = 255
            green = blue = int((1 + number) * 255)  # Increase other components less as it gets more negative
        elif number > 0:
            # Increase towards blue, full intensity at 1.0
            blue = 255
            red = green = int((1 - number) * 255)  # Increase other components less as it gets more positive
        else:
            # Exact 0, resulting in white
            red = green = blue = 255 
        
        # decide whether text is black or white depending on darkness of color
        text_hex = "#000000" if (red*0.299 + green*0.587 + blue*0.114) > 170 else "#ffffff"

        # Convert to hex, ensuring each component is 2 digits
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        
        return hex_code, text_hex
    
    if annotations is None:
        def get_label(name):
            return name
    else:
        def get_label(name):
            seq, feat = name.split(", ")
            if feat in annotations:
                component = feat.split('/')[0]
                component = feat.split('_')[0]
                return f'{seq}, {annotations[feat]} ({component})'
            return name

    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    nodes_by_submod = {
        'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
    }
    nodes_by_seqpos = defaultdict(list)
    nodes_by_layer = defaultdict(list)
    edgeset = set()

    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }
    edges['resid_-1'] = edges['embed']

    # add words to bottom of graph
    with G.subgraph(name=f'words') as subgraph:
        subgraph.attr(rank='same')
        prev_word = None
        for idx in range(length):
            word = words[idx]
            subgraph.node(word, shape='none', group=str(idx), fillcolor='transparent',
                          fontsize="30pt")
            if prev_word is not None:
                subgraph.edge(prev_word, word, style='invis', minlen="2")
            prev_word = word

    for layer in range(-1, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1 and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    seq_pos, basename = name.split(", ")
                    fillhex, texthex = to_hex(effect)
                    if name[-3:] == 'err':
                        subgraph.node(name, shape='triangle', group=seq_pos, width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), group=seq_pos, fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    
                    if len(nodes_by_seqpos[seq_pos]) == 0:
                        G.edge(words[int(seq_pos)], name, style='dotted', arrowhead='none', penwidth="1.5")
                        edgeset.add((words[int(seq_pos)], name))

                    nodes_by_seqpos[seq_pos].append(name)
                    nodes_by_layer[layer].append(name)

                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))
        
        # add edges to previous layer resid
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1: continue
            for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
                for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                    weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name('resid', layer-1, upstream_idx)
                        dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
                        edgeset.add((uname, dname))


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )
            edgeset.add((uname, dname))

    # if not os.path.exists(os.path.dirname(save_dir)):
    #     os.makedirs(os.path.dirname(save_dir))
    # G.render(save_dir, format='png', cleanup=True)
    return G