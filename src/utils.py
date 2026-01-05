from itertools import combinations
from collections import defaultdict
import numpy as np
import random
from typing import List, Dict, Set, Tuple, Iterable, Optional

from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

def matrix_to_edges(matrix, strings):
    edges = []
    n = len(strings)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i, j] == 1:
                edges.append((strings[i], strings[j]))
    return edges


def cluster_by_degree(matrix, strings):
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(strings)
    G.add_edges_from(matrix_to_edges(matrix, strings))
    
    # Compute degrees
    degree_map = dict(G.degree())
    
    # Group strings by degree
    grouped = defaultdict(list)
    for s, d in degree_map.items():
        grouped[d].append(s)
    
    # Sort by degree (ascending)
    sorted_groups = sorted(grouped.items())

    print("Degree -> Strings")
    for degree, group in sorted_groups:
        print(f"{degree:>2} -> {', '.join(sorted(group))}")

def plot_degree_histogram(matrix, strings, title):
    G = nx.Graph()
    G.add_nodes_from(strings)
    G.add_edges_from(matrix_to_edges(matrix, strings))

    degrees = [d for n, d in G.degree()]
    plt.figure()
    plt.hist(degrees, bins=range(0, max(degrees)+2), align='left', rwidth=0.8)
    plt.xlabel('Degree (Number of Equivalent Strings)')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def tsne_layout(G):
    nodes = list(G.nodes())
    
    # Features per node: [degree, clustering coefficient, eigenvector centrality]
    degree = np.array([G.degree(n) for n in nodes])
    clustering = np.array([nx.clustering(G, n) for n in nodes])
    eigenvector = np.array(list(nx.closeness_centrality(G).values()))
    
    features = np.stack([degree, clustering, eigenvector], axis=1)

    tsne = TSNE(n_components=2, perplexity=10, random_state=7)
    coords = tsne.fit_transform(features)

    pos = {node: tuple(coord) for node, coord in zip(nodes, coords)}
    return pos

def plot_clustered_graph(ax, matrix, strings, title, vmin=1, vmax=32, use_tsn=False):
    G = nx.Graph()
    G.add_nodes_from(strings)
    G.add_edges_from(matrix_to_edges(matrix, strings))
    # Compute connected components -> cluster IDs
    component_map = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for node in comp:
            component_map[node] = i
    # Node attributes
    degrees = dict(G.degree())
    colors = [degrees[node] for node in G.nodes()]
    nonzero_degrees = [deg for node, deg in degrees.items() if deg > 0]
    sizes = [400 + 10 * degrees[node]*0 for node in G.nodes()]  # degree-sensitive size
    # layout spring (force-directed)
    if use_tsn: 
        pos = tsne_layout(G)
    else:
        pos = nx.spring_layout(G, seed=17, k=2, iterations=80)
    
    # Plot on the given axis
    nodes = nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=colors,
        cmap=plt.cm.coolwarm,
        node_size=sizes,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)
    ax.set_title(title)
    ax.axis('off')
    return nodes

#Strings that co-occur within a sequence are assumed similar.
def extract_positive_pairs(sequences, window=2):
    pos_pairs = set()
    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i+1, min(i+1+window, len(seq))):
                pos_pairs.add((seq[i], seq[j]))
                pos_pairs.add((seq[j], seq[i]))
    return list(pos_pairs)
