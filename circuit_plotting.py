from graphviz import Digraph
import re
import os

def get_name(component, layer, idx):
    match idx:
        case (seq, feat):
            if feat == 32768: feat = 'res'
            return f'{seq}, {component}_{layer}/{feat}'
        case (feat,):
            if feat == 32768: feat = 'res'
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
        'embed': {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > node_threshold).nonzero()}
    }
    for layer in range(layers):
        for component in ['attn', 'mlp']:
            submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
            nodes_by_submod[f'{component}_{layer}'] = {
                tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
            }
        submod_nodes = nodes[f'resid_{layer+1}'].to_tensor()
        nodes_by_submod[f'resid_{layer+1}'] = {
            tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > node_threshold).nonzero()
        }

    for layer in range(layers):
        for component in ['attn', 'mlp', 'resid']:
            temp_layer = layer if component != 'resid' else layer + 1
            with G.subgraph(name=f'layer {temp_layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{temp_layer}'].items():
                    name = get_name(component, temp_layer, idx)
                    fillhex, texthex = to_hex(effect)
                    if name[-3:] == 'res':
                        subgraph.node(name, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    # if sequence position is present, separate nodes by sequence position
                    match idx:
                        case (seq, _):
                            subgraph.node(f'{component}_{temp_layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{temp_layer}_#{seq}_post', style='invis')
                            subgraph.edge(f'{component}_{temp_layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{temp_layer}_#{seq}_post', style='invis')
                            if max_seq_pos is None or seq > max_seq_pos:
                                max_seq_pos = seq

                if max_seq_pos is None: continue
                # make sure the auxiliary ordering nodes are in right order
                for seq in reversed(range(max_seq_pos+1)):
                    if f'{component}_{temp_layer}_#{seq}_pre' in ''.join(subgraph.body):
                        for seq_prev in range(seq):
                            if f'{component}_{temp_layer}_#{seq_prev}_post' in ''.join(subgraph.body):
                                subgraph.edge(f'{component}_{temp_layer}_#{seq_prev}_post', f'{component}_{temp_layer}_#{seq}_pre', style='invis')

        
        for component in ['attn', 'mlp']:
            for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
                for downstream_idx in nodes_by_submod[f'resid_{layer+1}'].keys():
                    weight = edges[f'{component}_{layer}'][f'resid_{layer+1}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                    if abs(weight) > edge_threshold:
                        uname = get_name(component, layer, upstream_idx)
                        dname = get_name('resid', layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )
        
        if layer > 0:
            # add edges from previous layer resid
            for component in ['attn', 'mlp', 'resid']:
                d_layer = layer if component != 'resid' else layer + 1
                for upstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
                    for downstream_idx in nodes_by_submod[f'{component}_{d_layer}'].keys():
                        weight = edges[f'resid_{layer}'][f'{component}_{d_layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                        if abs(weight) > edge_threshold:
                            uname = get_name('resid', layer, upstream_idx)
                            dname = get_name(component, d_layer, downstream_idx)
                            G.edge(
                                uname, dname,
                                penwidth=str(abs(weight) * pen_thickness),
                                color = 'red' if weight < 0 else 'blue'
                            )
        else:
            for component in ['attn', 'mlp', 'resid']:
                d_layer = layer if component != 'resid' else layer + 1
                for upstream_idx in nodes_by_submod['embed'].keys():
                    for downstream_idx in nodes_by_submod[f'{component}_{d_layer}'].keys():
                        weight = edges['embed'][f'{component}_{d_layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
                        if abs(weight) > edge_threshold:
                            uname = get_name('embed', 0, upstream_idx)
                            dname = get_name(component, d_layer, downstream_idx)
                            G.edge(
                                uname, dname,
                                penwidth=str(abs(weight) * pen_thickness),
                                color = 'red' if weight < 0 else 'blue'
                            )


    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers}'].keys():
        weight = edges[f'resid_{layers}']['y'][tuple(idx)].item()
        if abs(weight) > edge_threshold:
            name = get_name('resid', layers, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    G.render(save_dir, format='png', cleanup=True)
