### matrix 16 million rows, 60k columns 
### mostly sparse
import scipy as sp
import numpy as np
import time

rng = np.random.default_rng()
rows = 2_000_000
cols = 60_000
## density
# density = 1e-7  # extremely sparse - eigenvalue method faster
# density = 1e-5  # moderate density - good balance for testing
# density = 0.15  # 85% sparse (15% non-zero) - too dense for memory
density = 0.01  # 99% sparse (1% non-zero) - more realistic for scRNA-seq 
mat = sp.sparse.random(m=rows, n=cols, density=density, format='csr', rng=rng)
## mat.toarray() destroys sparsity


## mem usage
mem_usage = mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
print(f"Matrix created. Approximate memory usage: {mem_usage / 1e6:.2f} MB")
print(f"Matrix shape: {mat.shape}")
print(f"Number of non-zero elements: {mat.nnz}")

# convert to quad form
def quad_form(mat):
    return mat.T @ mat

def eigenvalue_svd(matrix, k=100):
    """
    Compute SVD via eigenvalue decomposition of covariance matrix.
    
    Args:
        matrix: Input sparse matrix (m x n)
        k: Number of components to compute
    
    Returns:
        sv: Singular values
        vt: V transpose matrix
    """
    print("Computing SVD via eigenvalue decomposition...")
    
    cov_mat = matrix.T @ matrix
    print(f"Covariance matrix shape: {cov_mat.shape}")
    print(f"Covariance matrix type: {type(cov_mat)}")

    if cov_mat.shape[0] <= 10000:  # Safety check for memory
        cov_mat_dense = cov_mat.toarray()
        eig_vals, eig_vecs = np.linalg.eig(cov_mat_dense)
        # For dense case, flip to get descending order
        eig_vals = np.flip(eig_vals)
        eig_vecs = np.flip(eig_vecs, axis=1)
    else:
        # For large matrices, use sparse eigenvalue solver
        from scipy.sparse.linalg import eigsh
        # Get top k eigenvalues/eigenvectors
        k = min(k, cov_mat.shape[0] - 1)  # number of components to compute
        eig_vals, eig_vecs = eigsh(cov_mat, k=k, which='LM')
        # eigsh already returns them in descending order for 'LM'

    # Compute singular values and V transpose
    sv = np.sqrt(np.abs(eig_vals))  # abs to handle numerical precision issues
    vt = eig_vecs.T 

    print(f"Number of computed eigenvalues: {len(eig_vals)}")
    print(f"Top 10 singular values: {sv[:10]}")
    print(f"VT shape: {vt.shape}")
    print(f"First few elements of VT[0]: {vt[0][:5] if len(vt[0]) >= 5 else vt[0]}")
    
    return sv, vt


def randomized_svd(matrix, k=100, n_oversamples=10, n_iter=2):
    """
    Randomized SVD using Halko-Martinsson-Tropp algorithm.
    
    Args:
        matrix: Input sparse matrix (m x n)
        k: Target rank (number of components)
        n_oversamples: Additional samples for better approximation
        n_iter: Number of power iterations for better accuracy
    
    Returns:
        U: Left singular vectors (m x k)
        s: Singular values (k,)
        Vt: Right singular vectors (k x n)
    """
    print("Computing SVD via Randomized SVD (Halko-Martinsson-Tropp)...")
    
    m, n = matrix.shape
    l = k + n_oversamples  # oversampled dimension
    
    # Step 1: Generate random test matrix
    Omega = rng.standard_normal((n, l))
    
    # Step 2: Form Y = A * Omega
    Y = matrix @ Omega
    
    # Step 3: Power iterations for better accuracy (optional)
    for _ in range(n_iter):
        Y = matrix @ (matrix.T @ Y)
    
    # Step 4: QR decomposition to get orthonormal basis
    Q, _ = np.linalg.qr(Y, mode='reduced')
    
    # Step 5: Form smaller matrix B = Q^T * A
    B = Q.T @ matrix
    
    # Step 6: SVD of smaller matrix B
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)
    
    # Step 7: Recover U = Q * U_tilde
    U = Q @ U_tilde
    
    # Truncate to desired rank k
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    
    print(f"RSVD completed. Shape: U={U.shape}, s={s.shape}, Vt={Vt.shape}")
    print(f"Top 10 singular values: {s[:10]}")
    
    return U, s, Vt


# Test eigenvalue decomposition method
print("\n" + "="*50)
print("EIGENVALUE DECOMPOSITION METHOD")
print("="*50)
start_time = time.time()
sv_eigen, vt_eigen = eigenvalue_svd(mat, k=100)
eigen_time = time.time() - start_time
print(f"Eigenvalue decomposition completed in {eigen_time:.2f} seconds")

# Test randomized SVD method  
print("\n" + "="*50)
print("RANDOMIZED SVD METHOD")
print("="*50)
start_time = time.time()
U_rsvd, s_rsvd, Vt_rsvd = randomized_svd(mat, k=100)
rsvd_time = time.time() - start_time
print(f"Randomized SVD completed in {rsvd_time:.2f} seconds")

# Compare results
print("\n" + "="*50)
print("COMPARISON OF METHODS")
print("="*50)

# Compare singular values
print(f"Eigenvalue method - top 10 singular values: {sv_eigen[:10]}")
print(f"RSVD method - top 10 singular values: {s_rsvd[:10]}")

# Compute relative error in singular values
min_len = min(len(sv_eigen), len(s_rsvd))
rel_error = np.abs(sv_eigen[:min_len] - s_rsvd[:min_len]) / sv_eigen[:min_len]
print(f"Relative error in singular values (mean): {np.mean(rel_error):.6f}")
print(f"Relative error in singular values (max): {np.max(rel_error):.6f}")

# Memory usage comparison
print(f"\nMemory usage:")
print(f"Eigenvalue method: Forms {cols}×{cols} covariance matrix")
print(f"RSVD method: Works directly on {mat.shape} matrix")

# Timing comparison
print(f"\nTiming comparison:")
print(f"Eigenvalue decomposition: {eigen_time:.2f} seconds")
print(f"Randomized SVD: {rsvd_time:.2f} seconds")
print(f"Speedup: {eigen_time/rsvd_time:.2f}x {'(RSVD faster)' if rsvd_time < eigen_time else '(Eigen faster)'}")

print(f"\nComplexity comparison:")
print(f"Eigenvalue SVD: O(n³) for eigendecomposition of {cols}×{cols} covariance matrix")
print(f"Randomized SVD: O(mnk) where m={rows}, n={cols}, k=100")


# Why Randomized SVD?
# (SNPs x Individuals)
# top Principal Components capture population structrure
# SVD is O(mn²)
# Randomized SVD is O(mnk) where k<<min(m,n)
## Assumption -- since the data is noisy, small approximation error is acceptable, and we get a massive speedup



# # Compute covariance matrix for analysis (reusing from eigenvalue method)
# print("\n" + "="*50)
# print("COVARIANCE MATRIX ANALYSIS")
# print("="*50)
# cov_mat = mat.T @ mat
# print(f"Recomputed covariance matrix for analysis: {cov_mat.shape}")

# # diagonal extraction - identifies variant strength and Hardy-Weinberg deviations
# mat_diag = cov_mat.diagonal()
# print(f"diagonal: shape={mat_diag.shape}, mean={mat_diag.mean():.6f}, std={mat_diag.std():.6f}")

# # Off-diagonal bands - captures linkage disequilibrium patterns for imputation and fine-mapping
# for k in range(-5, 6):
#     if k != 0:
#         diag_k = cov_mat.diagonal(k)
#         if len(diag_k) > 0:
#             print(f"  Diagonal {k:+2d}: length={len(diag_k)}, mean={diag_k.mean():.6f}, std={diag_k.std():.6f}")

# # 5. Pattern density analysis - detects structural variants and population stratification
# pattern_density = np.count_nonzero(mat_diag) / len(mat_diag)
# print(f"\nPattern density analysis:")
# print(f"Main diagonal density: {pattern_density:.6f} ({pattern_density*100:.2f}% non-zero)")
# for k in range(-3, 4):
#     if k != 0:
#         diag_k = cov_mat.diagonal(k)
#         if len(diag_k) > 0:
#             density_k = np.count_nonzero(diag_k) / len(diag_k)
#             print(f"Diagonal {k:+2d} density: {density_k:.6f} ({density_k*100:.2f}% non-zero)")

# # 4. Hotspot detection - finds GWAS hits and regulatory elements with high correlation
# threshold = np.percentile(mat_diag, 95)  # top 5%
# hotspot_indices = np.where(mat_diag > threshold)[0]
# print(f"\nHotspot detection (top 5%):")
# print(f"Found {len(hotspot_indices)} genomic hotspots")
# print(f"Threshold: {threshold:.6f}")
# print(f"Hotspot positions: {hotspot_indices[:10]}...")
# print(f"Hotspot values: {mat_diag[hotspot_indices[:10]]}")

# # 3. Block diagonal analysis - analyzes chromosomal regions and topological domains
# block_size = 1000
# n_blocks = min(10, cov_mat.shape[0] // block_size)
# print(f"\nBlock diagonal analysis (block_size={block_size}):")
# for i in range(n_blocks):
#     start_idx = i * block_size
#     end_idx = min((i + 1) * block_size, cov_mat.shape[0])
#     block_diag = cov_mat[start_idx:end_idx, start_idx:end_idx].diagonal()
#     print(f"  Block {i}: pos {start_idx}-{end_idx}, mean={block_diag.mean():.6f}, max={block_diag.max():.6f}")




