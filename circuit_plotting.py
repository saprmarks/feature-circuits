from graphviz import Digraph
import json
import torch as t


with open('feature_annotations.json', 'r') as f:
    feature_annotations = json.load(f)

def get_name(component, layer, idx):
    seq, feat = idx
    if feat == 32768: feat = 'res'
    return f'{seq}, {component}_{layer}/{feat}'

def get_label(name):
    seq, feat = name.split(', ')
    if feat in feature_annotations:
        component = feat.split('/')[0]
        component = feat.split('_')[0]
        return f'{seq}, {feature_annotations[feat]} ({component})'
    return name

def plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, pen_thickness=1):


    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    scale = max(abs(min_effect), abs(max_effect))

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
        
        # Convert to hex, ensuring each component is 2 digits
        hex_code = f'#{red:02X}{green:02X}{blue:02X}'
        
        return hex_code


    G = Digraph(name='Feature circuit')
    G.graph_attr.update(rankdir='BT', newrank='true')
    G.node_attr.update(shape="box", style="rounded")

    nodes_by_submod = {}
    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }

    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    if name[-3:] == 'res':
                        subgraph.node(name, shape='triangle', fillcolor=to_hex(effect), style='filled')
                    else:
                        subgraph.node(name, label=get_label(name), fillcolor=to_hex(effect), style='filled')
        
        for component in ['attn', 'mlp']:
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
        
        if layer > 0:
            # add edges to previous layer resid
            for component in ['attn', 'mlp', 'resid']:
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

    G.render('circuit', format='png', cleanup=True)