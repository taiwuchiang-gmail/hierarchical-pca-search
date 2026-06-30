import numpy as np
import time
import os
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# --- Data Handling Functions ---

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# --- Optimized Search Algorithm Implementations ---

def brute_force_1nn_l2(dataset, query_vector):
    """
    MODERN BASELINE: 
    Fully vectorized NumPy operation. It leverages underlying C/SIMD instructions.
    This replaces the slow Python `for` loop from the original script.
    """
    # Vectorized computation of squared Euclidean distances
    distances_squared = np.sum((dataset - query_vector) ** 2, axis=1)
    best_idx = np.argmin(distances_squared)
    return best_idx, distances_squared[best_idx]


def hierarchical_pca_1nn_l2(dataset, query_vector, component_levels):
    """
    IMPROVED ALGORITHM:
    Removes the "cheat" search radius. Automatically discovers the tightest 
    possible search radius on the fly using the most promising early candidate.
    """
    num_vectors, dim = dataset.shape
    candidate_indices = np.arange(num_vectors)
    pruning_stats = [num_vectors]

    # Initialize dynamic radius to infinity
    best_idx = -1
    best_dist_sq = float('inf')

    for i, num_components in enumerate(component_levels):
        if not candidate_indices.size:
            break
            
        query_coarse = query_vector[:num_components]
        dataset_coarse = dataset[candidate_indices, :num_components]

        # Calculate lower-bound distances
        distances_squared = np.sum((dataset_coarse - query_coarse) ** 2, axis=1)
        
        # --- SMART RADIUS INITIALIZATION ---
        # At the very first stage, find the candidate with the lowest 8D distance.
        # Compute its full 128D distance immediately! This gives us a mathematically
        # guaranteed, highly accurate pruning radius for all subsequent points without cheating.
        if i == 0:
            # Get indices of the top 50 closest points in 8D
            top_50_local_idx = np.argpartition(distances_squared, 50)[:50]
            top_50_global_idx = candidate_indices[top_50_local_idx]

            # Compute exact 128D distances for these 50 points
            exact_dists_50 = np.sum((dataset[top_50_global_idx] - query_vector) ** 2, axis=1)

            # Use the absolute best one as our pruning radius!
            best_dist_sq = np.min(exact_dists_50)
            best_idx = top_50_global_idx[np.argmin(exact_dists_50)]

        # Prune candidates using the dynamically acquired best distance
        passing_mask = distances_squared < best_dist_sq
        candidate_indices = candidate_indices[passing_mask]
        pruning_stats.append(len(candidate_indices))

    # Final Search: Full 128D evaluation on the tiny fraction of surviving candidates
    if candidate_indices.size > 0:
        final_dists_sq = np.sum((dataset[candidate_indices] - query_vector) ** 2, axis=1)
        min_final_idx_local = np.argmin(final_dists_sq)
        
        if final_dists_sq[min_final_idx_local] < best_dist_sq:
            best_dist_sq = final_dists_sq[min_final_idx_local]
            best_idx = candidate_indices[min_final_idx_local]
            
    return best_idx, best_dist_sq, pruning_stats


def visualize_pruning_effectiveness(avg_stats, component_levels):
    """Generates a bar chart showing the average pruning at each stage."""
    labels = ['Total Dataset'] + [f'After {n}D PCA' for n in component_levels] + ['Final Full-D Search']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_stats, color='#4C72B0', edgecolor='black')
    plt.yscale('log')
    plt.ylabel('Avg Number of Candidate Vectors (Log Scale)', fontsize=12)
    plt.title('Hierarchical Pruning Effectiveness on SIFT1M', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval):,}', va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig('figures/pruning_effectiveness.png', dpi=300)
    print("\nGenerated 'figures/pruning_effectiveness.png' for the manuscript.")

def run_benchmark_multiple_queries():
    sift_dir = './sift' # Assumes downloaded from your original script
    if not os.path.exists(sift_dir):
        print("Please run the original script once to download SIFT1M.")
        return

    pca_model_path = 'pca_sift.joblib'
    transformed_dataset_path = 'sift_base_pca.npy'

    print("Loading pre-computed PCA model and dataset...")
    base_vectors_pca = np.load(transformed_dataset_path)
    pca = joblib.load(pca_model_path)
    query_vectors_raw = fvecs_read(os.path.join(sift_dir, 'sift_query.fvecs'))
    
    # Let's benchmark over 100 queries for statistical significance
    num_queries_to_test = 100
    print(f"\n--- Running Benchmark over {num_queries_to_test} Queries ---")
    
    time_brute_total = 0
    time_hierarchical_total = 0
    component_hierarchy = [8, 16, 32, 64]
    
    all_pruning_stats = []
    
    for q_idx in range(num_queries_to_test):
        query_vector_raw = query_vectors_raw[q_idx]
        query_vector_pca = pca.transform(query_vector_raw.reshape(1, -1))[0]
        
        # 1. Improved Hierarchical Search
        start = time.perf_counter()
        h_idx, h_dist, h_stats = hierarchical_pca_1nn_l2(base_vectors_pca, query_vector_pca, component_hierarchy)
        time_hierarchical_total += (time.perf_counter() - start)
        
        # Append the final stage count (the ones that go to 128D)
        h_stats.append(h_stats[-1])
        all_pruning_stats.append(h_stats)
        
        # 2. Vectorized Brute Force
        start = time.perf_counter()
        b_idx, b_dist = brute_force_1nn_l2(base_vectors_pca, query_vector_pca)
        time_brute_total += (time.perf_counter() - start)
        
        # Verify correctness
        assert h_idx == b_idx, f"Query {q_idx} Failed: Mismatch in nearest neighbor!"
        
        # Print progress every 20 queries
        if (q_idx + 1) % 20 == 0:
            print(f"Processed {q_idx + 1}/{num_queries_to_test} queries...")

    avg_hierarchical = (time_hierarchical_total / num_queries_to_test) * 1000
    avg_brute = (time_brute_total / num_queries_to_test) * 1000
    
    print("\n--- Final Results (Average per Query) ---")
    print(f"Vectorized Brute Force: {avg_brute:.2f} ms")
    print(f"Hierarchical PCA Search: {avg_hierarchical:.2f} ms")
    print(f"Remaining candidates after [8, 16, 32, 64] filters: {h_stats[-2]} (for the last query)")
    
    # Calculate averages across all queries
    avg_pruning_stats = np.mean(all_pruning_stats, axis=0)
    
    os.makedirs('figures', exist_ok=True)
    visualize_pruning_effectiveness(avg_pruning_stats, component_hierarchy)

if __name__ == '__main__':
    run_benchmark_multiple_queries()
