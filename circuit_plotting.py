from graphviz import Digraph
from collections import defaultdict
import re
import os

"""
class GridCircuit:
    def __init__(self, nodes, edges, node_threshold, edge_threshold, annotations,
                 model_config, seq_len):
        self.circuit_nodes = nodes
        self.circuit_edges = edges
        self.model_config = model_config
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.num_layers = model_config.num_hidden_layers
        self.sequence_length
        self.annotations = annotations

        self.nodes_by_layer = None
    
    # compute width and height of entire figure, and width of sequence position blocks
    def compute_figure_dimensions(self):
        #
        # Returns (figure_width, figure_height, sequence_position_widths)
        #
        max_layer_width = 0
        seqpos_widths = [0] * self.sequence_length
        height = 1  # start at 1 to account for `y`

        nodes_by_submod = {
            'resid_-1' : {tuple(idx.tolist()) : self.circuit_nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (self.circuit_nodes['embed'].to_tensor().abs() > self.node_threshold).nonzero()}
        }

        for layer in range(-1, self.num_layers+1):
            resid_width, mlpattn_width = 0, 0
            mlpattn_seqpos_len, resid_seqpos_len = defaultdict(int), defaultdict(int)
            if layer == -1 or layer == self.num_layers:
                component_list = ['resid']
            else:
                component_list = ['attn', 'mlp', 'resid']
            for component in component_list:
                submod_nodes = self.circuit_nodes[f'{component}_{layer}'].to_tensor()
                nodes_by_submod[f'{component}_{layer}'] = {
                    tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > self.node_threshold).nonzero()
                }
                submod_num_nodes = len(list(nodes_by_submod[f'{component}_{layer}'].keys()))
                if component in ("mlp", "attn"):
                    mlpattn_width += submod_num_nodes
                else:
                    resid_width += submod_num_nodes
                
                for idx in nodes_by_submod[f'{component}_{layer}']:
                    seqpos, _ = idx.split(", ")
                    seqpos = int(seqpos)
                    if component in ("mlp", "attn"):
                        mlpattn_seqpos_len[seqpos] += 1
                    else:
                        resid_seqpos_len[seqpos] += 1    

            max_layer_width = max(max_layer_width, resid_width, mlpattn_width)        
            if resid_width > 0:
                height += 1
            if mlpattn_width > 0:
                height += 1
            
            for seqpos in mlpattn_seqpos_len:
                seqpos_widths[seqpos] = max(seqpos_widths[seqpos], mlpattn_seqpos_len[seqpos])
            for seqpos in resid_seqpos_len:
                seqpos_widths[seqpos] = max(seqpos_widths[seqpos], resid_seqpos_len[seqpos])
        
        return max_layer_width, height, seqpos_widths


    def to_hex(self, number, scale):
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
    

    def produce_plot(self):
        # 1. draw all nodes, including empty nodes. draw invisible edges to make the grid
        # get min and max node effects
        min_effect = min([v.to_tensor().min() for n, v in self.circuit_nodes.items() if n != 'y'])
        max_effect = max([v.to_tensor().max() for n, v in self.circuit_nodes.items() if n != 'y'])
        scale = max(abs(min_effect), abs(max_effect))

        if self.annotations is None:
            def get_label(name):
                return name
        else:
            def get_label(name):
                match name.split(', '):
                    case seq, feat:
                        if feat in self.annotations:
                            component = feat.split('/')[0]
                            component = feat.split('_')[0]
                            return f'{seq}, {self.annotations[feat]} ({component})'
                        return name
                    case [feat]:
                        if feat in self.annotations:
                            component = feat.split('/')[0]
                            component = feat.split('_')[0]
                            return f'{self.annotations[feat]} ({component})'

        G = Digraph(name='Feature circuit')
        G.graph_attr.update(rankdir='BT', newrank='true')
        G.node_attr.update(shape="box", style="rounded")

        fig_width, fig_height, seqpos_widths = self.compute_figure_dimensions()

        nodes_by_submod = {
            'resid_-1' : {tuple(idx.tolist()) : self.circuit_nodes['embed'].to_tensor()[tuple(idx)].item() for \
                          idx in (self.circuit_nodes['embed'].to_tensor().abs() > self.node_threshold).nonzero()}
        }
        for layer in range(self.model_config.num_hidden_layers):
            for component in ['attn', 'mlp', 'resid']:
                submod_nodes = self.circuit_nodes[f'{component}_{layer}'].to_tensor()
                nodes_by_submod[f'{component}_{layer}'] = {
                    tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > self.node_threshold).nonzero()
                }
        self.circuit_edges['resid_-1'] = self.circuit_edges['embed']

        for layer in range(-1, layers):
            for component in ['attn', 'mlp', 'resid']:
                if layer == -1 and component != 'resid': continue
                with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                    subgraph.attr(rank='same')
                    max_seq_pos = None
                    # sort nodes by sequence position
                    prev_node = None
                    for curr_seq_pos in range(self.seq_len):
                        actual_nodes_at_seqpos = 0
                        for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                            name = get_name(component, layer, idx)
                            if idx[0] != curr_seq_pos:
                                continue
                            actual_nodes_at_seqpos += 1
                            fillhex, texthex = self.to_hex(effect, scale)
                            if name[-3:] == 'err':
                                subgraph.node(name, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                            fillcolor=fillhex, style='filled', fontcolor=texthex)
                            else:
                                subgraph.node(name, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                            style='filled')
                            if prev_node is not None:
                                G.edge(prev_node, name)
                            prev_node = name
                            # if sequence position is present, separate nodes by sequence position
                            # subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
                            # subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
                            # if max_seq_pos is None or seq > max_seq_pos:
                            #     max_seq_pos = seq
                    if actual_nodes_at_seqpos < seqpos_widths[curr_seq_pos]:
                        # add invisible nodes until we reach the right width
                        for _ in range(actual_nodes_at_seqpos, seqpos_widths[curr_seq_pos]):
                            fake_name = f"{layer}_{}_{curr_seq_pos}"
                            subgraph.node()



        # 3. draw actual edges
    
"""

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
    # if component == 'resid': layer += 1
    match idx:
        case (seq, feat):
            if feat == 32768: feat = 'ε'
            if layer == -1: return f'{seq}, embed/{feat}'
            return f'{seq}, {component}_{layer}/{feat}'
        case (feat,):
            if feat == 32768: feat = 'ε'
            if layer == -1: return f'embed/{feat}'
            return f'{component}_{layer}/{feat}'
        case _: raise ValueError(f"Invalid idx: {idx}")

def locate_merge_name(name, merges):
    for merge_name in merges:
        if name in merges[merge_name]:
            return merge_name


def plot_circuit_merge(nodes, edges, merges, length=6, example_text=None,
                       layers=6, node_threshold=0.1, edge_threshold=0.01,
                       pen_thickness=1, annotations=None, save_dir='circuit'):
    merges_seen = defaultdict(list)
    all_merge_nodes = []
    for merge_names in merges:
        all_merge_nodes.extend(merges[merge_names])
    merge_name_effects = defaultdict(float)
    merge_name_inweights = defaultdict(lambda: defaultdict(float))
    merge_name_outweights = defaultdict(lambda: defaultdict(float))
    
    # get min and max node effects
    min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    max_effect = max([v.to_tensor().sum() for n, v in nodes.items() if n != 'y'])
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
    G.graph_attr.update(rankdir='BT', newrank='true', concentrate="true")
    G.node_attr.update(shape="box", style="rounded")

    nodes_by_seqpos = defaultdict(list)
    nodes_by_layer = defaultdict(list)
    edgeset = set()

    # rename embed to resid_-1
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

    # add words to bottom of graph
    with G.subgraph(name=f'words') as subgraph:
        subgraph.attr(rank='same')
        prev_word = None
        for idx in range(length):
            word = words[idx]
            subgraph.node(word, shape='none', group=str(idx), fillcolor='transparent',
                          fontsize="30pt")
            if prev_word is not None:
                subgraph.edge(prev_word, word, style='invis')
            prev_word = word
    
    for layer in range(-1, layers):
        for component in ['attn', 'mlp', 'resid']:
            if layer == -1 and component != 'resid': continue
            with G.subgraph(name=f'layer {layer} {component}') as subgraph:
                subgraph.attr(rank='same')
                max_seq_pos = None
                for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
                    name = get_name(component, layer, idx)
                    if name in all_merge_nodes:
                        original_name = name
                        name = locate_merge_name(name, merges)
                        merge_name_effects[name] += effect
                        effect = merge_name_effects[name]

                    seq_pos, basename = name.split(", ")
                    fillhex, texthex = to_hex(effect)

                    if name[-1:].endswith('ε'):
                        subgraph.node(name, shape='triangle', width="1.6", height="0.8", fixedsize="true",
                                      fillcolor=fillhex, style='filled', fontcolor=texthex)
                    else:
                        subgraph.node(name, label=get_label(name), fillcolor=fillhex, fontcolor=texthex,
                                      style='filled')
                    
                    if len(nodes_by_seqpos[seq_pos]) == 0:
                        G.edge(words[int(seq_pos)], name, style='dotted', arrowhead='none', penwidth="1.5")
                        edgeset.add((words[int(seq_pos)], name))
                    nodes_by_seqpos[seq_pos].append(name)
                    nodes_by_layer[layer].append(name)

                    if name in merges:
                        if original_name not in merges_seen[name]:
                            merges_seen[name].append(original_name)

                        if len(merges_seen[name]) == len(merges[name]):
                            # if sequence position is present, separate nodes by sequence position
                            match idx:
                                case (seq, _):
                                    subgraph.node(f'{component}_{layer}_#{seq}_pre', width="0.1", style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', width="0.1", style='invis')
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
                    uname = get_name(component, layer, upstream_idx)
                    dname = get_name('resid', layer, downstream_idx)
                    if dname in all_merge_nodes:
                        dname = locate_merge_name(dname, merges)
                    if uname in all_merge_nodes:
                        uname = locate_merge_name(uname, merges)
                        merge_name_outweights[uname][dname] += weight
                        weight = merge_name_outweights[uname][dname]
                    if uname == dname:
                        continue

                    if abs(weight) > edge_threshold:
                        # uname = get_name(component, layer, upstream_idx)
                        # dname = get_name('resid', layer, downstream_idx)
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
                    uname = get_name('resid', layer-1, upstream_idx)
                    dname = get_name(component, layer, downstream_idx)
                    if dname in all_merge_nodes:
                        dname = locate_merge_name(dname, merges)
                    if uname in all_merge_nodes:
                        uname = locate_merge_name(uname, merges)
                        merge_name_outweights[uname][dname] += weight
                        weight = merge_name_outweights[uname][dname]
                    if uname == dname:
                        continue

                    if abs(weight) > edge_threshold:
                        # uname = get_name('resid', layer-1, upstream_idx)
                        # dname = get_name(component, layer, downstream_idx)
                        G.edge(
                            uname, dname,
                            penwidth=str(abs(weight) * pen_thickness),
                            color = 'red' if weight < 0 else 'blue'
                        )

    # the cherry on top
    G.node('y', shape='diamond')
    for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
        weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
        name = get_name('resid', layers-1, idx)
        if name in all_merge_nodes:
            name = locate_merge_name(name, merges)
            merge_name_outweights[name]["y"] += weight
            weight = merge_name_outweights[name]["y"]

        if abs(weight) > edge_threshold:
            # name = get_name('resid', layers-1, idx)
            G.edge(
                name, 'y',
                penwidth=str(abs(weight) * pen_thickness),
                color = 'red' if weight < 0 else 'blue'
            )

    # if not os.path.exists(os.path.dirname(save_dir)):
    #     os.makedirs(os.path.dirname(save_dir))
    # G.render(save_dir, format='png', cleanup=True)
    return G


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

    # rename embed to resid_-1
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
                    if name[-1:].endswith('ε'):
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
                component = component.split('_')[0]
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
                    if name[-1:] == 'ε':
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