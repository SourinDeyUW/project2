"""
Polyhedra Analysis Pipeline for CIF Files

This script extracts polyhedra from CIF files and performs comprehensive analysis including:
1. Polyhedra extraction and visualization
2. Polyhedron sharing analysis (face, edge, point connections)
3. Graph construction and topological embedding
4. Final embedding preparation for machine learning

Usage:
    python extract_polyhedra_analysis.py <cif_filename>
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from itertools import combinations

# Import functions from existing files
from view_polyhedra import count_and_visualize_polyhedra_interactive
from draw_graphs import (
    remove_overlapping_polyhedra,
    get_polyhedron_sharing_pairs_verbose,
    draw_polyhedron_sharing_graph
)

def get_polyhedron_type(coordination_number):
    """
    Get a polyhedron type name based on coordination number.
    """
    polyhedron_types = {
        1: "Mono",
        2: "Linear",
        3: "Triangular",
        4: "Tetrahedral",
        5: "Trigonal bipyramidal",
        6: "Octahedral",
        7: "Pentagonal bipyramidal",
        8: "Cubic/Square antiprismatic",
        9: "Tricapped trigonal prismatic",
        10: "Bicapped square antiprismatic",
        11: "Pentagonal antiprismatic",
        12: "Icosahedral/Cuboctahedral"
    }
    return polyhedron_types.get(coordination_number, f"CN-{coordination_number}")

def edge_type_histogram_embedding(G, bins=10):
    """
    Extract graph topological features and convert into embedding.
    
    Args:
        G: NetworkX MultiGraph representing polyhedron connectivity
        bins: Number of bins for histogram features
    
    Returns:
        numpy.ndarray: Feature vector of length 4*bins
    """
    face_tri_deg = {n: 0 for n in G.nodes}
    face_quad_deg = {n: 0 for n in G.nodes}
    edge_deg = {n: 0 for n in G.nodes}
    point_deg = {n: 0 for n in G.nodes}

    for u, v, data in G.edges(data=True):
        etype = data.get("type")
        if etype == "face-tri":
            face_tri_deg[u] += 1
            face_tri_deg[v] += 1
        elif etype == "face-quad":
            face_quad_deg[u] += 1
            face_quad_deg[v] += 1
        elif etype == "edge":
            edge_deg[u] += 1
            edge_deg[v] += 1
        elif etype == "point":
            point_deg[u] += 1
            point_deg[v] += 1

    # Compute histograms
    tri_hist, _ = np.histogram(list(face_tri_deg.values()), bins=bins, range=(0, bins))
    quad_hist, _ = np.histogram(list(face_quad_deg.values()), bins=bins, range=(0, bins))
    edge_hist, _ = np.histogram(list(edge_deg.values()), bins=bins, range=(0, bins))
    point_hist, _ = np.histogram(list(point_deg.values()), bins=bins, range=(0, bins))

    # Normalize
    tri_feat = tri_hist / tri_hist.sum() if tri_hist.sum() > 0 else np.zeros(bins)
    quad_feat = quad_hist / quad_hist.sum() if quad_hist.sum() > 0 else np.zeros(bins)
    edge_feat = edge_hist / edge_hist.sum() if edge_hist.sum() > 0 else np.zeros(bins)
    point_feat = point_hist / point_hist.sum() if point_hist.sum() > 0 else np.zeros(bins)

    return np.concatenate([tri_feat, quad_feat, edge_feat, point_feat])  # total = 4 * bins


def extract_polyhedra_from_cif(cif_filename, cifs_folder="cifs", remove_overlaps=True, distance_threshold=1.5):
    """
    Extract polyhedra data from a CIF file.
    
    Args:
        cif_filename: Name of the CIF file (e.g., "mp-757245.cif")
        cifs_folder: Folder containing CIF files
        remove_overlaps: Whether to remove overlapping polyhedra
        distance_threshold: Distance threshold for overlap removal
    
    Returns:
        tuple: (polyhedra_counts, cn_counts, polyhedra_data)
    """
    print(f"Processing CIF file: {cif_filename}")
    
    # Create a custom version of the function that uses our cifs folder
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    from scipy.spatial import ConvexHull
    
    # Load the structure from our cifs folder
    cif_path = os.path.join(cifs_folder, cif_filename)
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    structure = Structure.from_file(cif_path)
    
    # Use CrystalNN to determine coordination environments
    nn_analyzer = CrystalNN()
    
    # Collect coordination numbers, polyhedra vertices, and angle distortions
    coordination_numbers = []
    polyhedra_data = []
    
    cation_sites = []
    ignore = ["O", "S", "N", "F", "Cl", "Br", "I", "H"]  # ignored these as center atoms
    for site_idx in range(len(structure)):
        site = structure[site_idx]
        if site.specie.symbol not in ignore:
            cation_sites.append(site_idx)
    
    # Analyze each cation site
    for site_idx in cation_sites:
        nn_info = nn_analyzer.get_nn_info(structure, site_idx)
        cn = len(nn_info)
        coordination_numbers.append(cn)
        
        center = structure[site_idx].coords
        vertices = [info['site'].coords for info in nn_info]
        vertex_elements = [info['site'].specie.symbol for info in nn_info]
        vertex_indices = [info['site_index'] for info in nn_info]
        
        # Get polyhedron type
        polyhedron_type = get_polyhedron_type(cn)
    
        # Calculate angle distortion
        if cn >= 3:
            angles = []
            for i in range(cn):
                for j in range(i + 1, cn):
                    v1 = vertices[i] - center
                    v2 = vertices[j] - center
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    if angle < 150:
                        angles.append(angle)
            angles = np.array(angles)
            if cn == 4:
                ideal_angle = 109.47
            elif cn == 6:
                ideal_angle = 90.0
            elif cn == 8:
                ideal_angle = 70.5
            else:
                ideal_angle = np.mean(angles)
            angle_distortion = np.sqrt(np.mean((angles - ideal_angle) ** 2))
        else:
            angle_distortion = None
    
        # Face indices
        face_indices = []
        if len(vertices) >= 4:
            try:
                hull = ConvexHull(vertices)
                for simplex in hull.simplices:
                    face = sorted([vertex_indices[i] for i in simplex])
                    face_indices.append(face)
            except:
                pass  # hull may fail on degenerate configs
    
        # Store info
        polyhedra_data.append({
            'center': center,
            'vertices': vertices,
            'vertex_indices': vertex_indices,
            'faces': face_indices,
            'cn': cn,
            'type': polyhedron_type,
            'element': structure[site_idx].specie.symbol,
            'angle_distortion': angle_distortion,
            'center_index': site_idx,
            've_elements': vertex_elements
        })
    
    # Count polyhedron types
    cn_counts = Counter(coordination_numbers)
    polyhedra_counts = {get_polyhedron_type(cn): count for cn, count in cn_counts.items()}
    
    print(f"Initial number of polyhedra: {len(polyhedra_data)}")
    print(f"Coordination number distribution: {cn_counts}")
    print(f"Polyhedra type distribution: {polyhedra_counts}")
    
    # Remove overlapping polyhedra if requested
    if remove_overlaps and len(polyhedra_data) > 1:
        cleaned_data, removed_indices = remove_overlapping_polyhedra(
            polyhedra_data, 
            distance_threshold=distance_threshold
        )
        print(f"Removed {len(removed_indices)} overlapping polyhedra")
        print(f"Final number of polyhedra: {len(cleaned_data)}")
        polyhedra_data = cleaned_data
    
    return polyhedra_counts, cn_counts, polyhedra_data


def analyze_polyhedron_connectivity(polyhedra_data):
    """
    Analyze polyhedron connectivity and extract sharing pairs.
    
    Args:
        polyhedra_data: List of polyhedra dictionaries
    
    Returns:
        tuple: (face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs)
    """
    print("\nAnalyzing polyhedron connectivity...")
    
    # Get sharing pairs using existing function
    face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs = get_polyhedron_sharing_pairs_verbose(polyhedra_data)
    
    print(f"Triangle face-sharing pairs: {len(face_tri_pairs)}")
    print(f"Quad face-sharing pairs: {len(face_quad_pairs)}")
    print(f"Edge-sharing pairs: {len(edge_pairs)}")
    print(f"Point-sharing pairs: {len(point_pairs)}")
    
    # Print some examples
    if face_tri_pairs:
        print(f"Example triangle face-sharing: {list(face_tri_pairs.items())[:2]}")
    if face_quad_pairs:
        print(f"Example quad face-sharing: {list(face_quad_pairs.items())[:2]}")
    if edge_pairs:
        print(f"Example edge-sharing: {list(edge_pairs.items())[:2]}")
    if point_pairs:
        print(f"Example point-sharing: {list(point_pairs.items())[:2]}")
    
    return face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs


def create_connectivity_graph(face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs):
    """
    Create a NetworkX graph representing polyhedron connectivity.
    
    Args:
        face_tri_pairs: Dictionary of triangle face-sharing pairs
        face_quad_pairs: Dictionary of quad face-sharing pairs
        edge_pairs: Dictionary of edge-sharing pairs
        point_pairs: Dictionary of point-sharing pairs
    
    Returns:
        networkx.MultiGraph: Graph representing polyhedron connectivity
    """
    print("\nCreating connectivity graph...")
    
    # Create graph using existing function
    G = draw_polyhedron_sharing_graph(face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs)
    
    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    
    # Analyze edge types
    edge_types = defaultdict(int)
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] += 1
    
    print(f"Edge type distribution: {dict(edge_types)}")
    
    return G


def generate_embedding(G, bins=10):
    """
    Generate embedding from connectivity graph.
    
    Args:
        G: NetworkX MultiGraph
        bins: Number of bins for histogram features
    
    Returns:
        numpy.ndarray: Embedding vector
    """
    print(f"\nGenerating embedding with {bins} bins...")
    
    embedding = edge_type_histogram_embedding(G, bins=bins)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"Embedding sum: {embedding.sum():.4f}")
    
    return embedding


def visualize_graph(G, save_path=None, show_plot=False):
    """
    Visualize the connectivity graph.
    
    Args:
        G: NetworkX MultiGraph
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot (default: False)
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph to visualize")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create layout
    if G.number_of_nodes() > 1:
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = {list(G.nodes())[0]: (0, 0)}
    
    # Define colors and styles for different edge types
    edge_styles = {
        'face-tri': {'color': 'blue', 'style': '--', 'width': 2, 'label': 'Triangle Face'},
        'face-quad': {'color': 'blue', 'style': '-', 'width': 3, 'label': 'Quad Face'},
        'edge': {'color': 'red', 'style': '-.', 'width': 2, 'label': 'Edge'},
        'point': {'color': 'gray', 'style': ':', 'width': 1, 'label': 'Point'}
    }
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges by type
    drawn_types = set()
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'unknown')
        if edge_type in edge_styles:
            style = edge_styles[edge_type]
            # Only add to legend once per type
            label = style['label'] if edge_type not in drawn_types else None
            drawn_types.add(edge_type)
            
            nx.draw_networkx_edges(G, pos, [(u, v)], 
                                 edge_color=style['color'],
                                 style=style['style'],
                                 width=style['width'],
                                 alpha=0.7,
                                 label=label)
    
    plt.title("Polyhedron Connectivity Graph", fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory


def save_results(cif_filename, polyhedra_data, embedding, G, output_dir="output"):
    """
    Save analysis results to files.
    
    Args:
        cif_filename: Original CIF filename
        polyhedra_data: Polyhedra analysis data
        embedding: Generated embedding
        G: Connectivity graph
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(cif_filename)[0]
    
    # Save embedding
    embedding_path = os.path.join(output_dir, f"{base_name}_embedding.npy")
    np.save(embedding_path, embedding)
    print(f"Embedding saved to: {embedding_path}")
    
    # Save graph with different formats
    graph_pkl_path = os.path.join(output_dir, f"{base_name}_graph.pkl")
    graph_gml_path = os.path.join(output_dir, f"{base_name}_graph.gml")
    
    # Save as pickle (preserves all data types)
    import pickle
    with open(graph_pkl_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph (pickle) saved to: {graph_pkl_path}")
    
    # Convert graph for GML export (strings only)
    G_gml = G.copy()
    for u, v, key, data in G_gml.edges(keys=True, data=True):
        for attr_name, attr_value in data.items():
            if not isinstance(attr_value, str):
                G_gml[u][v][key][attr_name] = str(attr_value)
    
    try:
        nx.write_gml(G_gml, graph_gml_path)
        print(f"Graph (GML) saved to: {graph_gml_path}")
    except Exception as e:
        print(f"Warning: Could not save GML format: {e}")
        print(f"Graph data saved as pickle only: {graph_pkl_path}")
    
    # Save polyhedra summary
    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Polyhedra Analysis Summary for {cif_filename}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of polyhedra: {len(polyhedra_data)}\n")
        f.write(f"Graph nodes: {G.number_of_nodes()}\n")
        f.write(f"Graph edges: {G.number_of_edges()}\n")
        f.write(f"Embedding dimensions: {len(embedding)}\n\n")
        
        # Coordination numbers
        cn_dist = Counter([p['cn'] for p in polyhedra_data])
        f.write("Coordination number distribution:\n")
        for cn, count in sorted(cn_dist.items()):
            f.write(f"  CN {cn}: {count} polyhedra\n")
        f.write("\n")
        
        # Polyhedra types
        type_dist = Counter([p['type'] for p in polyhedra_data])
        f.write("Polyhedra type distribution:\n")
        for ptype, count in sorted(type_dist.items()):
            f.write(f"  {ptype}: {count}\n")
        f.write("\n")
        
        # Edge types
        edge_types = defaultdict(int)
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] += 1
        
        f.write("Graph edge type distribution:\n")
        for etype, count in sorted(edge_types.items()):
            f.write(f"  {etype}: {count}\n")
    
    print(f"Summary saved to: {summary_path}")


def load_reference_dataset(dataset_path, query_cif_filename=None):
    """
    Load the reference dataset from pickle file containing NetworkX graphs.
    Optionally filter by prototype if query filename and dataset.csv are provided.
    
    Args:
        dataset_path: Path to the polyhedron_graphs3.pkl file
        query_cif_filename: Name of the query CIF file (without 'cifs/' prefix)
    
    Returns:
        tuple: (embeddings, labels) or (None, None) if failed
    """
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data type: {type(data)}")
        print(f"Number of structures in dataset: {len(data)}")
        
        # Extract first few keys to show structure
        sample_keys = list(data.keys())[:5]
        print(f"Sample keys: {sample_keys}")
        
        # Filter by prototype if query filename and dataset.csv are available
        filtered_data = data
        if query_cif_filename:
            try:
                # Read the dataset.csv to get prototype information
                df = pd.read_csv('dataset.csv')
                print(f"Dataset CSV loaded with {len(df)} entries")
                
                # Add 'cifs/' prefix to query filename to match the dataset
                query_key = f"cifs/{query_cif_filename}"
                print(f"Looking for query key: {query_key}")
                
                # Find the prototype of the query structure
                query_prototype = None
                
                if 'mp_id' in df.columns and 'prototype' in df.columns:
                    # The mp_id column already contains the full path like 'cifs/mp-xxxx.cif'
                    query_row = df[df['mp_id'] == query_key]
                    if not query_row.empty:
                        query_prototype = query_row['prototype'].iloc[0]
                        print(f"Query structure {query_key} has prototype: {query_prototype}")
                        
                        # Filter the dataset to only include structures with the same prototype
                        same_prototype_df = df[df['prototype'] == query_prototype]
                        same_prototype_keys = same_prototype_df['mp_id'].tolist()
                        
                        # Filter the data to only include structures with same prototype
                        filtered_data = {key: value for key, value in data.items() 
                                       if key in same_prototype_keys}
                        
                        print(f"Filtered to {len(filtered_data)} structures with prototype '{query_prototype}'")
                    else:
                        print(f"Warning: Query structure {query_key} not found in dataset.csv")
                        print("Using full dataset for similarity search")
                else:
                    print("Warning: 'mp_id' or 'prototype' columns not found in dataset.csv")
                    print("Using full dataset for similarity search")
                    
            except FileNotFoundError:
                print("Warning: dataset.csv not found. Using full dataset for similarity search")
            except Exception as e:
                print(f"Warning: Error processing dataset.csv: {e}")
                print("Using full dataset for similarity search")
        
        # Generate embeddings using the same function as step2
        X = []
        valid_mp_ids = []
        
        print("Generating embeddings for reference dataset...")
        for mp_id, G in filtered_data.items():
            try:
                emb = edge_type_histogram_embedding(G, bins=10)
                if not np.isnan(emb).any():
                    X.append(emb)
                    valid_mp_ids.append(mp_id)
            except Exception as e:
                print(f"Warning: Could not generate embedding for {mp_id}: {e}")
                continue
        
        if len(X) == 0:
            print("No valid embeddings generated from reference dataset")
            return None, None
            
        X = np.array(X)
        print(f"Generated {len(X)} embeddings with shape {X.shape}")
        
        return X, valid_mp_ids
        
    except Exception as e:
        print(f"Error loading reference dataset: {e}")
        return None, None


def edge_type_histogram_embedding(G, bins=10):
    """
    Generate embedding from graph using edge type histograms.
    This is the same function from step2.ipynb.
    
    Args:
        G: NetworkX graph with edge type annotations
        bins: Number of bins for histograms
    
    Returns:
        numpy.ndarray: Concatenated histogram features
    """
    face_tri_deg = {n: 0 for n in G.nodes}
    face_quad_deg = {n: 0 for n in G.nodes}
    edge_deg = {n: 0 for n in G.nodes}
    point_deg = {n: 0 for n in G.nodes}

    for u, v, data in G.edges(data=True):
        etype = data.get("type")
        if etype == "face-tri":
            face_tri_deg[u] += 1
            face_tri_deg[v] += 1
        elif etype == "face-quad":
            face_quad_deg[u] += 1
            face_quad_deg[v] += 1
        elif etype == "edge":
            edge_deg[u] += 1
            edge_deg[v] += 1
        elif etype == "point":
            point_deg[u] += 1
            point_deg[v] += 1

    # Compute histograms
    tri_hist, _ = np.histogram(list(face_tri_deg.values()), bins=bins, range=(0, bins))
    quad_hist, _ = np.histogram(list(face_quad_deg.values()), bins=bins, range=(0, bins))
    edge_hist, _ = np.histogram(list(edge_deg.values()), bins=bins, range=(0, bins))
    point_hist, _ = np.histogram(list(point_deg.values()), bins=bins, range=(0, bins))

    # Normalize
    tri_feat = tri_hist / tri_hist.sum() if tri_hist.sum() > 0 else np.zeros(bins)
    quad_feat = quad_hist / quad_hist.sum() if quad_hist.sum() > 0 else np.zeros(bins)
    edge_feat = edge_hist / edge_hist.sum() if edge_hist.sum() > 0 else np.zeros(bins)
    point_feat = point_hist / point_hist.sum() if point_hist.sum() > 0 else np.zeros(bins)

    return np.concatenate([tri_feat, quad_feat, edge_feat, point_feat])  # total = 4 * bins


def find_similar_structures(query_embedding, reference_embeddings, reference_labels, k=5):
    """
    Find k most similar structures using t-SNE proximity.
    
    Args:
        query_embedding: Embedding vector for the query structure
        reference_embeddings: Array of reference embeddings
        reference_labels: Labels/names for reference structures
        k: Number of similar structures to return
    
    Returns:
        list: List of (distance, label) tuples, sorted by similarity (closest first)
    """
    from sklearn.manifold import TSNE
    
    # Ensure embeddings are numpy arrays
    query_embedding = np.array(query_embedding)
    reference_embeddings = np.array(reference_embeddings)
    
    # Handle 1D query embedding
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Reference embeddings shape: {reference_embeddings.shape}")
    
    # Handle case where query and reference have different dimensions
    if query_embedding.shape[1] != reference_embeddings.shape[1]:
        min_dim = min(query_embedding.shape[1], reference_embeddings.shape[1])
        query_embedding = query_embedding[:, :min_dim]
        reference_embeddings = reference_embeddings[:, :min_dim]
        print(f"Warning: Dimension mismatch, using first {min_dim} features")
    
    try:
        # Combine query and reference embeddings for t-SNE
        combined_embeddings = np.vstack([query_embedding, reference_embeddings])
        print(f"Combined embeddings shape: {combined_embeddings.shape}")
        
        # Apply t-SNE to the combined data
        print("Applying t-SNE to find structure similarities...")
        perplexity = min(10, len(combined_embeddings) - 1)  # Ensure perplexity is valid
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded_points = tsne.fit_transform(combined_embeddings)
        
        # The first point is our query
        query_point = embedded_points[0]
        reference_points = embedded_points[1:]
        
        # Calculate Euclidean distances in t-SNE space
        distances = []
        for i, ref_point in enumerate(reference_points):
            distance = np.linalg.norm(query_point - ref_point)
            label = reference_labels[i] if i < len(reference_labels) else f"structure_{i}"
            distances.append((distance, label))
        
        # Sort by distance (closest first) and return top k
        distances.sort(key=lambda x: x[0])
        k = min(k, len(distances))  # Don't ask for more than available
        
        return distances[:k]
        
    except Exception as e:
        print(f"Error in t-SNE similarity calculation: {e}")
        print("Falling back to cosine similarity...")
        
        # Fallback to cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
        
        query_norm = normalize(query_embedding)
        ref_norm = normalize(reference_embeddings)
        
        similarities = cosine_similarity(query_norm, ref_norm)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            label = reference_labels[idx] if idx < len(reference_labels) else f"structure_{idx}"
            # Convert similarity to distance (1 - similarity for consistency)
            distance = 1 - similarity_score
            results.append((distance, label))
        
        return results


def analyze_with_similarity(cif_filename, dataset_path, k=5, show_plots=False):
    """
    Analyze a CIF file and find similar structures from reference dataset.
    
    Args:
        cif_filename: Name of the CIF file to analyze
        dataset_path: Path to the reference dataset pickle file
        k: Number of similar structures to find
        show_plots: Whether to display plots
    
    Returns:
        dict: Analysis results including similar structures
    """
    print(f"Analyzing {cif_filename} with similarity search...")
    print("=" * 60)
    
    # First, perform the standard analysis
    result = main(cif_filename, show_plots=show_plots)
    
    if result is None:
        print("Analysis failed, cannot perform similarity search")
        return None
    
    # Load reference dataset
    print(f"\nLoading reference dataset from: {dataset_path}")
    ref_embeddings, ref_labels = load_reference_dataset(dataset_path, cif_filename)
    
    if ref_embeddings is None:
        print("Failed to load reference dataset")
        return result
    
    print(f"Reference dataset loaded: {len(ref_embeddings)} structures")
    
    # Find similar structures
    query_embedding = result['embedding']
    similar_structures = find_similar_structures(
        query_embedding, ref_embeddings, ref_labels, k=k
    )
    
    # Read dataset.csv to get formulas for the similar structures
    try:
        df = pd.read_csv('dataset.csv')
        formula_dict = dict(zip(df['mp_id'], df['reduced_formula']))
        
        # Get the query structure's formula
        query_key = f"cifs/{cif_filename}"
        query_formula = formula_dict.get(query_key, "Unknown")
        print(f"Query structure: {cif_filename} ({query_formula})")
        
    except Exception as e:
        print(f"Warning: Could not load formulas from dataset.csv: {e}")
        formula_dict = {}
        query_formula = "Unknown"
    
    # Display results with formulas
    print(f"\nTop {k} most similar structures to {cif_filename} ({query_formula}):")
    print("-" * 70)
    for i, (distance, label) in enumerate(similar_structures, 1):
        formula = formula_dict.get(label, "Unknown")
        print(f"{i:2d}. {label:<30} {formula:<15} (t-SNE distance: {distance:.4f})")
    
    # Add similarity results to the return dict
    result['similar_structures'] = similar_structures
    result['reference_dataset_size'] = len(ref_embeddings)
    
    # Save similarity results
    os.makedirs("output", exist_ok=True)
    base_name = os.path.splitext(cif_filename)[0]
    similarity_path = f"output/{base_name}_similarity.txt"
    
    with open(similarity_path, 'w') as f:
        f.write(f"Similarity Analysis for {cif_filename}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Query embedding shape: {query_embedding.shape}\n")
        f.write(f"Reference dataset size: {len(ref_embeddings)}\n\n")
        f.write(f"Top {k} most similar structures (by t-SNE proximity):\n")
        f.write("-" * 60 + "\n")
        for i, (distance, label) in enumerate(similar_structures, 1):
            formula = formula_dict.get(label, "Unknown")
            f.write(f"{i:2d}. {label:<30} {formula:<15} (distance: {distance:.4f})\n")
    
    print(f"Similarity results saved to: {similarity_path}")
    
    return result


def main(cif_filename, show_plots=False):
    """
    Main pipeline for polyhedra analysis.
    
    Args:
        cif_filename: Name of the CIF file to analyze
        show_plots: Whether to display plots (default: False)
    """
    print(f"Starting polyhedra analysis pipeline for: {cif_filename}")
    print("=" * 60)
    
    try:
        # Step 1: Extract polyhedra from CIF file
        polyhedra_counts, cn_counts, polyhedra_data = extract_polyhedra_from_cif(cif_filename)
        
        if not polyhedra_data:
            print("No polyhedra found in the structure!")
            return
        
        # Step 2: Analyze polyhedron connectivity
        face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs = analyze_polyhedron_connectivity(polyhedra_data)
        
        # Step 3: Create connectivity graph
        G = create_connectivity_graph(face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs)
        
        # Step 4: Generate embedding
        embedding = generate_embedding(G, bins=10)
        
        # Step 5: Visualize graph
        graph_viz_path = f"output/{os.path.splitext(cif_filename)[0]}_graph.png"
        visualize_graph(G, save_path=graph_viz_path, show_plot=show_plots)
        
        # Step 6: Save results
        save_results(cif_filename, polyhedra_data, embedding, G)
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Results saved in 'output' directory")
        
        return {
            'polyhedra_data': polyhedra_data,
            'connectivity_graph': G,
            'embedding': embedding,
            'face_tri_pairs': face_tri_pairs,
            'face_quad_pairs': face_quad_pairs,
            'edge_pairs': edge_pairs,
            'point_pairs': point_pairs
        }
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_multiple_cifs(cif_folder="cifs", show_plots=False):
    """
    Analyze multiple CIF files in a folder.
    
    Args:
        cif_folder: Folder containing CIF files
        show_plots: Whether to display plots (default: False)
    
    Returns:
        dict: Results for each CIF file
    """
    if not os.path.exists(cif_folder):
        print(f"CIF folder '{cif_folder}' not found!")
        return {}
    
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith('.cif')]
    
    if not cif_files:
        print(f"No CIF files found in '{cif_folder}'!")
        return {}
    
    print(f"Found {len(cif_files)} CIF files to analyze")
    
    all_results = {}
    all_embeddings = []
    
    for cif_file in cif_files:
        print(f"\n{'='*60}")
        print(f"Processing {cif_file} ({cif_files.index(cif_file)+1}/{len(cif_files)})")
        
        result = main(cif_file, show_plots=show_plots)
        if result is not None:
            all_results[cif_file] = result
            all_embeddings.append(result['embedding'])
    
    # Save combined embeddings
    if all_embeddings:
        embeddings_matrix = np.array(all_embeddings)
        combined_path = "output/all_embeddings.npy"
        np.save(combined_path, embeddings_matrix)
        print(f"\nCombined embeddings saved to: {combined_path}")
        print(f"Shape: {embeddings_matrix.shape}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze polyhedra in CIF files')
    parser.add_argument('cif_file', nargs='?', help='CIF file to analyze (optional)')
    parser.add_argument('--show-plots', action='store_true', 
                       help='Display interactive plots (default: False)')
    parser.add_argument('--plots', action='store_true', 
                       help='Display interactive plots (short form)')
    parser.add_argument('--find-similar', type=str, metavar='DATASET_PATH',
                       help='Find similar structures using reference dataset (provide path to .pkl file)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of most similar structures to find (default: 5)')
    
    args = parser.parse_args()
    show_plots = args.show_plots or args.plots
    
    if args.cif_file:
        # Single file analysis
        cif_filename = args.cif_file
        if not cif_filename.endswith('.cif'):
            cif_filename += '.cif'
        
        if args.find_similar:
            # Perform similarity analysis
            analyze_with_similarity(
                cif_filename, 
                args.find_similar, 
                k=args.top_k,
                show_plots=show_plots
            )
        else:
            # Standard analysis
            main(cif_filename, show_plots=show_plots)
    else:
        if args.find_similar:
            print("Error: --find-similar requires a specific CIF file. Please provide a CIF file name.")
            sys.exit(1)
        
        # Analyze all CIF files in the cifs folder
        print("No specific CIF file provided. Analyzing all CIF files in 'cifs' folder...")
        if show_plots:
            print("Note: Plots will be displayed for each structure (this may take longer)")
        analyze_multiple_cifs(show_plots=show_plots)
