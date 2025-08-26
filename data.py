import scanpy as sc
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import time
from rsvd import randomized_svd, eigenvalue_svd

# Load raw feature bc matrix
# data_dir = "/Users/sud/Documents/repos/mat/raw_feature_bc_matrix"
data_dir = "/Users/sud/Documents/repos/mat/filtered_feature_bc_matrix"

adata = sc.read_10x_mtx(
    data_dir,
    var_names='gene_symbols',  # Use gene symbols for variable names
    cache=True                 # Cache the data for faster subsequent loading
)

# Print the AnnData object to see its contents
print("Original data shape:", adata.X.shape)
print("Data type:", type(adata.X))
print("AnnData object:")
print(adata)

# Convert to dense matrix if it's sparse (for SVD)
if hasattr(adata.X, 'todense'):
    print("\nConverting sparse matrix to dense for SVD...")
    X_dense = adata.X.todense()
else:
    X_dense = adata.X

print("Dense matrix shape:", X_dense.shape)

# SVD Comparison: SciPy vs Randomized SVD
print("\n" + "="*60)
print("SVD COMPARISON: SciPy vs Randomized SVD")
print("="*60)

# Parameters for comparison
k = min(50, min(adata.X.shape) - 1)  # Number of singular values to compute
print(f"Computing top {k} singular values for comparison")

# Method 1: SciPy truncated SVD
print("\n" + "-"*40)
print("METHOD 1: SciPy Truncated SVD")
print("-"*40)
scipy_start_time = time.time()
try:
    U_scipy, s_scipy, Vt_scipy = svds(adata.X, k=k)
    # svds returns singular values in ascending order, so reverse them
    U_scipy = U_scipy[:, ::-1]
    s_scipy = s_scipy[::-1]
    Vt_scipy = Vt_scipy[::-1, :]
    scipy_time = time.time() - scipy_start_time
    scipy_success = True
    print(f"SciPy SVD completed successfully in {scipy_time:.3f} seconds")
    print(f"U shape: {U_scipy.shape}, s shape: {s_scipy.shape}, Vt shape: {Vt_scipy.shape}")
    print(f"Top 10 singular values: {s_scipy[:10]}")
except Exception as e:
    scipy_time = time.time() - scipy_start_time
    scipy_success = False
    print(f"SciPy SVD failed after {scipy_time:.3f} seconds: {e}")

# Method 2: Randomized SVD (from rsvd.py)
print("\n" + "-"*40)
print("METHOD 2: Randomized SVD (Halko-Martinsson-Tropp)")
print("-"*40)
rsvd_start_time = time.time()
try:
    U_rsvd, s_rsvd, Vt_rsvd = randomized_svd(adata.X, k=k, n_oversamples=10, n_iter=2)
    rsvd_time = time.time() - rsvd_start_time
    rsvd_success = True
    print(f"Randomized SVD completed successfully in {rsvd_time:.3f} seconds")
    print(f"U shape: {U_rsvd.shape}, s shape: {s_rsvd.shape}, Vt shape: {Vt_rsvd.shape}")
    print(f"Top 10 singular values: {s_rsvd[:10]}")
except Exception as e:
    rsvd_time = time.time() - rsvd_start_time
    rsvd_success = False
    print(f"Randomized SVD failed after {rsvd_time:.3f} seconds: {e}")

# Method 3: Eigenvalue SVD (from rsvd.py) - for comparison
print("\n" + "-"*40)
print("METHOD 3: Eigenvalue Decomposition SVD")
print("-"*40)
eigen_start_time = time.time()
try:
    sv_eigen, vt_eigen = eigenvalue_svd(adata.X, k=k)
    eigen_time = time.time() - eigen_start_time
    eigen_success = True
    print(f"Eigenvalue SVD completed successfully in {eigen_time:.3f} seconds")
    print(f"sv shape: {sv_eigen.shape}, vt shape: {vt_eigen.shape}")
    print(f"Top 10 singular values: {sv_eigen[:10]}")
except Exception as e:
    eigen_time = time.time() - eigen_start_time
    eigen_success = False
    print(f"Eigenvalue SVD failed after {eigen_time:.3f} seconds: {e}")

# Comparison Results
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

# Timing comparison
print("\nTiming Results:")
if scipy_success:
    print(f"  SciPy SVD:        {scipy_time:.3f} seconds")
if rsvd_success:
    print(f"  Randomized SVD:   {rsvd_time:.3f} seconds")
if eigen_success:
    print(f"  Eigenvalue SVD:   {eigen_time:.3f} seconds")

if scipy_success and rsvd_success:
    speedup = scipy_time / rsvd_time
    print(f"\nSpeedup: Randomized SVD is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than SciPy SVD")

# Accuracy comparison
if scipy_success and rsvd_success:
    print("\nAccuracy Comparison (SciPy vs Randomized SVD):")
    min_len = min(len(s_scipy), len(s_rsvd))
    rel_error = np.abs(s_scipy[:min_len] - s_rsvd[:min_len]) / (s_scipy[:min_len] + 1e-10)  # Add small epsilon to avoid division by zero
    print(f"  Relative error in singular values:")
    print(f"    Mean: {np.mean(rel_error):.6f}")
    print(f"    Max:  {np.max(rel_error):.6f}")
    print(f"    Std:  {np.std(rel_error):.6f}")

if scipy_success and eigen_success:
    print("\nAccuracy Comparison (SciPy vs Eigenvalue SVD):")
    min_len = min(len(s_scipy), len(sv_eigen))
    rel_error_eigen = np.abs(s_scipy[:min_len] - sv_eigen[:min_len]) / (s_scipy[:min_len] + 1e-10)
    print(f"  Relative error in singular values:")
    print(f"    Mean: {np.mean(rel_error_eigen):.6f}")
    print(f"    Max:  {np.max(rel_error_eigen):.6f}")
    print(f"    Std:  {np.std(rel_error_eigen):.6f}")

# Memory and complexity analysis
print("\nComplexity Analysis:")
m, n = adata.X.shape
print(f"  Matrix dimensions: {m} × {n}")
print(f"  SciPy SVD:        O(min(m,n)³) - uses ARPACK/Lanczos")
print(f"  Randomized SVD:   O(mnk) where k={k}")
print(f"  Eigenvalue SVD:   O(n³) for {n}×{n} covariance matrix")

print("\nMemory Usage:")
print(f"  SciPy SVD:        Works directly on sparse matrix")
print(f"  Randomized SVD:   Works directly on sparse matrix")  
print(f"  Eigenvalue SVD:   Creates {n}×{n} covariance matrix")

print("\n" + "="*60)