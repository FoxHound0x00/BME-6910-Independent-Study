## Matrix Properties
- **Dimensions**: SNP (500K-10M × 10K-1M), Expression (20K-60K × 100-100K), Hi-C (1M-10M × 1M-10M)
- **Sparsity**: SNP (90-99%), Expression (60-80%), Hi-C (99.9%), Protein networks (99.99%)
- **Structure**: Block diagonal (chromosomes), banded (LD/TADs), long-range correlations (regulatory)

## Mathematical Theory of Randomized SVD

### Overview

Randomized SVD provides an efficient approximation to the Singular Value Decomposition by exploiting randomized sampling techniques. For a matrix $A \in \mathbb{R}^{m \times n}$, the goal is to compute an approximate rank-$k$ factorization $A \approx \tilde{U}\tilde{\Sigma}\tilde{V}^T$ where $k \ll \min(m,n)$.

### Problem Formulation

Given:
- Matrix $A \in \mathbb{R}^{m \times n}$
- Target rank $k \ll \min(m,n)$
- Oversampling parameter $p \geq 0$ (typically $p = 5$ to $10$)

Find: Approximate rank-$k$ SVD $A \approx \tilde{U}_k\tilde{\Sigma}_k\tilde{V}_k^T$

### Stage 1: Range Finding

**Objective**: Find an orthonormal basis $Q \in \mathbb{R}^{m \times (k+p)}$ such that $A \approx QQ^TA$.

#### Basic Algorithm (Halko-Martinsson-Tropp)

1. **Random Sampling Matrix**: Generate $\Omega \in \mathbb{R}^{n \times (k+p)}$ with i.i.d. Gaussian entries:
   $$\Omega_{ij} \sim \mathcal{N}(0, 1)$$

2. **Sample the Range**: Compute the sample matrix:
   $$Y = A\Omega \in \mathbb{R}^{m \times (k+p)}$$

3. **Orthogonalization**: Compute orthonormal basis via QR decomposition:
   $$Y = QR \quad \text{where } Q \in \mathbb{R}^{m \times (k+p)}, \, Q^TQ = I$$

#### Power Iteration Enhancement

For matrices with slowly decaying singular values, apply $q$ power iterations:

1. **Initialize**: $Y^{(0)} = A\Omega$
2. **Power Iteration**: For $j = 0, 1, \ldots, q-1$:
   $$Z^{(j)} = A^T Y^{(j)}$$
   $$Y^{(j+1)} = A Z^{(j)}$$
3. **Orthogonalize**: $Y^{(q)} = QR$

**Computational Cost**: $O(mnk + qmn)$ operations.

#### Subspace Iteration (Advanced)

For better numerical stability with power iterations:

1. **Initialize**: $Y^{(0)} = A\Omega$, compute $Y^{(0)} = Q^{(0)}R^{(0)}$
2. **Iterate**: For $j = 0, 1, \ldots, q-1$:
   - $Z^{(j)} = A^T Q^{(j)}$, compute $Z^{(j)} = \tilde{Q}^{(j)}\tilde{R}^{(j)}$
   - $Y^{(j+1)} = A \tilde{Q}^{(j)}$, compute $Y^{(j+1)} = Q^{(j+1)}R^{(j+1)}$
3. **Output**: $Q = Q^{(q)}$

### Stage 2: Direct SVD

**Objective**: Compute SVD of the reduced matrix $B = Q^TA$.

1. **Form Reduced Matrix**:
   $$B = Q^TA \in \mathbb{R}^{(k+p) \times n}$$

2. **Compute Small SVD**:
   $$B = \tilde{U}_B \tilde{\Sigma} \tilde{V}^T$$
   where $\tilde{U}_B \in \mathbb{R}^{(k+p) \times (k+p)}$, $\tilde{\Sigma} \in \mathbb{R}^{(k+p) \times (k+p)}$, $\tilde{V} \in \mathbb{R}^{n \times (k+p)}$.

3. **Reconstruct Left Singular Vectors**:
   $$\tilde{U} = Q\tilde{U}_B \in \mathbb{R}^{m \times (k+p)}$$

4. **Extract Rank-k Approximation**:
   $$A \approx \tilde{U}_k \tilde{\Sigma}_k \tilde{V}_k^T$$
   where subscript $k$ denotes the first $k$ columns/rows.

### Error Analysis

#### Deterministic Error Bounds

For the best rank-$k$ approximation $A_k = U_k\Sigma_k V_k^T$:

$$\|A - QQ^TA\|_2 \leq \|A - A_k\|_2 + \sigma_{k+1}$$

where $\sigma_{k+1}$ is the $(k+1)$-th singular value of $A$.

#### Probabilistic Error Bounds

**Theorem (Halko et al.)**: Let $A \in \mathbb{R}^{m \times n}$, $k \geq 2$, $p \geq 2$, and $\Omega$ be a standard Gaussian matrix. Then:

$$\mathbb{E}\left[\|A - QQ^TA\|_2\right] \leq \left(1 + \frac{4\sqrt{k+p}}{p-1}\right) \sigma_{k+1}$$

**High Probability Bound**: With probability at least $1 - 6e^{-p}$:

$$\|A - QQ^TA\|_2 \leq \left(1 + \frac{9\sqrt{k+p}}{p}\right) \sigma_{k+1}$$

#### Frobenius Norm Bounds

$$\mathbb{E}\left[\|A - QQ^TA\|_F^2\right] \leq \|A - A_k\|_F^2 + \frac{k}{p-1}\sum_{j=k+1}^{\min(m,n)} \sigma_j^2$$

### Power Iteration Analysis

For matrices with singular value decay $\sigma_j \leq C j^{-\alpha}$ (where $\alpha > 1$), $q$ power iterations yield:

$$\|A - Q^{(q)}(Q^{(q)})^TA\|_2 \lesssim C k^{-\alpha} \left(\frac{\sigma_{k+1}}{\sigma_k}\right)^{2q}$$

**Optimal Choice**: $q = O(\log(\sigma_k/\sigma_{k+1}))$ power iterations suffice.

### Computational Complexity

| Operation | Standard SVD | Randomized SVD | Speedup |
|-----------|-------------|----------------|---------|
| **Basic** | $O(\min(mn^2, m^2n))$ | $O(mnk + (m+n)k^2)$ | $\sim k/\min(m,n)$ |
| **With $q$ Power Iterations** | $O(\min(mn^2, m^2n))$ | $O(qmn + (m+n)k^2)$ | $\sim 1/q$ when $k \ll \min(m,n)$ |
| **Memory** | $O(mn)$ | $O((m+n)k)$ | $\sim k/\min(m,n)$ |

### Genomic Applications

#### 1. Population Structure (PCA)
For genotype matrix $G \in \{0,1,2\}^{n \times p}$ ($n$ individuals, $p$ SNPs):
- **Full PCA**: $O(np^2)$ when $n \ll p$
- **Randomized PCA**: $O(npk + n^2k)$ for $k$ components
- **Typical**: $n \sim 10^6$, $p \sim 10^7$, $k \sim 20$ → $1000\times$ speedup

#### 2. Expression Matrix Factorization
For expression matrix $E \in \mathbb{R}^{g \times s}$ ($g$ genes, $s$ samples):
- **Cell type signatures**: $E \approx WH$ where $W$ contains gene loadings
- **Batch correction**: Remove technical components via $E - \tilde{U}_{\text{tech}}\tilde{U}_{\text{tech}}^T E$

#### 3. Hi-C Contact Matrices
For contact matrix $C \in \mathbb{R}^{b \times b}$ ($b$ genomic bins):
- **Compartment analysis**: Top eigenvectors reveal A/B compartments
- **TAD detection**: Local eigenvalue structure identifies boundaries
- **Multi-resolution**: Hierarchical randomized SVD for nested structure

### Implementation Considerations

#### Numerical Stability
1. **Orthogonalization**: Use modified Gram-Schmidt or Householder QR
2. **Power Iteration**: Orthogonalize at each step to prevent loss of orthogonality
3. **Oversampling**: Use $p \geq 5$ to ensure numerical stability

#### Parallelization
1. **Matrix-Vector Products**: $A\Omega$ and $A^T Y$ are embarrassingly parallel
2. **QR Decomposition**: Use blocked algorithms (LAPACK DGEQRF)
3. **Small SVD**: Standard algorithms for $(k+p) \times (k+p)$ matrices

#### Memory Optimization
1. **Streaming**: Process matrix in blocks if $A$ doesn't fit in memory
2. **In-place Operations**: Reuse storage for intermediate results
3. **Sparse Matrices**: Exploit sparsity in genomic data (typically 90-99% sparse)

### Advanced Variants

#### Single-Pass Algorithm
When matrix access is expensive (e.g., streaming data):
$$Y = A\Omega, \quad Z = A^T\Omega$$
Then solve for $Q$ such that $Y = QR$ and compute $B = Q^T Z^T$.

#### Block Krylov Methods
For matrices with clustered singular values:
1. Generate multiple random matrices $\Omega_1, \ldots, \Omega_s$
2. Compute $Y_i = A\Omega_i$ for each block
3. Orthogonalize the combined space $[Y_1, \ldots, Y_s]$

#### Adaptive Rank Selection
Automatically determine rank $k$:
1. Start with initial guess $k_0$
2. Monitor singular value decay: $\sigma_i/\sigma_1 < \epsilon$
3. Adaptively increase $k$ until convergence criterion met

## Key Analysis Tasks
1. **GWAS**: Disease association mapping
2. **PCA**: Population structure correction  
3. **LD Analysis**: Linkage patterns, imputation
4. **eQTL**: Gene regulation mapping
5. **Structural Variants**: CNV/inversion detection
6. **Hi-C**: 3D genome organization

## Decomposition Methods

1. **Standard Eigenvalue**: [`np.linalg.eig()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) - `A = QΛQ^T`
   - **Symmetric**: `scipy.linalg.eigh()` - Hermitian/real symmetric matrices
   - **General**: `np.linalg.eig()` - Non-symmetric matrices, complex eigenvalues
   - **Genomic applications**: 
     * Kinship matrix eigendecomposition (relatedness estimation)
     * Covariance matrix PCA (population structure)
     * Correlation matrix analysis (co-expression networks)

2. **SVD**: [`np.linalg.svd()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) - `A = UΣV^T`
   - **Full SVD**: `np.linalg.svd(full_matrices=True)` - Complete decomposition
   - **Economy SVD**: `np.linalg.svd(full_matrices=False)` - Memory efficient
   - **Genomic applications**:
     * Gene expression matrix factorization (cell type signatures)
     * Genotype imputation (missing data completion)
     * Batch effect correction (technical variation removal)

3. **Sparse Eigenvalue**: [`scipy.sparse.linalg.eigsh()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) - Partial eigendecomposition
   - **Symmetric**: `eigsh(A, k, which='LM')` - Largest magnitude eigenvalues
   - **General**: `eigs(A, k, which='LR')` - Largest real part eigenvalues
   - **Genomic applications**:
     * Large-scale GWAS kinship matrices (population structure)
     * Hi-C contact matrix analysis (chromatin compartments)
     * Protein interaction networks (community detection)

4. **Randomized SVD**: [`sklearn.decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) - Approximate decomposition
   - **Halko algorithm**: `fbpca.pca()` - Fast randomized PCA
   - **Sketching methods**: Random projections for dimension reduction
   - **Genomic applications**:
     * Biobank-scale PCA (UK Biobank, All of Us)
     * Single-cell RNA-seq dimensionality reduction
     * Population genetics with millions of SNPs

5. **Generalized Eigenvalue**: [`scipy.linalg.eigh(A, B)`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html) - `Ax = λBx`
   - **Standard form**: Transform to `A' = B^{-1/2}AB^{-1/2}`
   - **Cholesky decomposition**: `scipy.linalg.cholesky()` for positive definite B
   - **Genomic applications**:
     * Linear mixed models (REML/ML estimation)
     * Heritability estimation (GCTA, BOLT-LMM)
     * Multi-trait GWAS (correlated phenotypes)

6. **Block Diagonal**: Manual blocking - Exploit matrix structure
   - **Chromosome blocking**: Analyze each chromosome separately
   - **Population blocking**: Separate analysis by ancestry groups  
   - **Temporal blocking**: Time-series genomic data
   - **Genomic applications**:
     * Chromosome-wise LD analysis
     * Population-stratified GWAS
     * Multi-tissue eQTL mapping

7. **Banded Matrix**: [`scipy.linalg.eigh_tridiagonal()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html) - Exploit band structure
   - **Tridiagonal**: `eigh_tridiagonal(d, e)` - Main + off-diagonal
   - **Pentadiagonal**: Custom LAPACK routines
   - **Genomic applications**:
     * Markov chain models (haplotype phasing)
     * Sequence alignment scoring matrices
     * Local LD decay modeling

8. **Iterative (Lanczos/Arnoldi)**: [`scipy.sparse.linalg.eigsh()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) - Krylov subspace methods
   - **Lanczos**: Symmetric matrices, tridiagonalization
   - **Arnoldi**: Non-symmetric matrices, Hessenberg form
   - **Genomic applications**:
     * Gene regulatory network analysis
     * Pathway enrichment eigenvalue problems
     * Phylogenetic tree construction

9. **Matrix-Free**: [`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) - Implicit matrix operations
   - **Custom matvec**: Define `A @ x` without storing A
   - **Kernel methods**: Gram matrix operations
   - **Genomic applications**:
     * Kernel GWAS (implicit kinship matrices)
     * Large-scale covariance estimation
     * Memory-efficient PCA for biobanks

10. **Hierarchical**: Custom algorithms - Multi-scale decomposition
    - **Recursive blocking**: Nested matrix partitioning
    - **Multilevel methods**: Coarse-to-fine analysis
    - **Genomic applications**:
      * Phylogenetic analysis (species trees)
      * Population structure (nested ancestry)
      * Multi-resolution Hi-C analysis

11. **NMF**: [`sklearn.decomposition.NMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) - `A ≈ WH`, W,H ≥ 0
    - **Multiplicative updates**: Lee & Seung algorithm
    - **Coordinate descent**: Faster convergence
    - **Genomic applications**:
      * Cell type deconvolution (bulk RNA-seq)
      * Mutational signature analysis (SBS, DBS, ID signatures)
      * Copy number profile factorization

12. **Tensor Decomposition**: [`tensorly.decomposition`](https://tensorly.org/stable/modules/api.html#module-tensorly.decomposition) - Multi-way data
    - **CP Decomposition**: `tensorly.decomposition.parafac()` - Canonical polyadic (rank-1 tensors)
    - **Tucker Decomposition**: `tensorly.decomposition.tucker()` - Higher-order SVD
    - **Tensor Train**: `tensorly.decomposition.tensor_train()` - Sequential matrix products
    - **Genomic applications**: 
      * SNP × Individual × Population tensors (3-way population genetics)
      * Gene × Sample × Condition × Time (4-way expression dynamics) 
      * Genomic position × Cell type × Individual (3-way single-cell eQTL)
      * Variant × Trait × Population (3-way GWAS meta-analysis)

## Key References

### Mathematical
- **Halko et al. (2011)** - Randomized matrix decompositions - [DOI: 10.1137/090771806](https://doi.org/10.1137/090771806)
  * *Use for*: Randomized SVD (#4), biobank-scale PCA, single-cell RNA-seq dimensionality reduction
- **Martinsson & Tropp (2020)** - Randomized numerical linear algebra - [DOI: 10.1017/S0962492920000021](https://doi.org/10.1017/S0962492920000021)
  * *Use for*: Randomized SVD (#4), sketching methods, approximate decompositions for large genomic datasets
- **Candès & Recht (2009)** - Matrix completion - [DOI: 10.1007/s10208-009-9045-5](https://doi.org/10.1007/s10208-009-9045-5)
  * *Use for*: Matrix-free methods (#9), genotype imputation, missing data completion in expression matrices
- **Kolda & Bader (2009)** - Tensor decompositions and applications - [DOI: 10.1137/07070111X](https://doi.org/10.1137/07070111X)
  * *Use for*: Tensor decomposition (#12), multi-way genomic data analysis, interaction effects

### Genomics
- **Price et al. (2006)** - PCA for population stratification - [DOI: 10.1038/ng1847](https://doi.org/10.1038/ng1847)
  * *Use for*: Standard eigenvalue (#1), sparse eigenvalue (#3), GWAS population structure correction
- **Yang et al. (2011)** - GCTA toolkit - [DOI: 10.1016/j.ajhg.2010.11.011](https://doi.org/10.1016/j.ajhg.2010.11.011)
  * *Use for*: Generalized eigenvalue (#5), heritability estimation, kinship matrix decomposition
- **Gabriel et al. (2002)** - Haplotype blocks - [DOI: 10.1126/science.1069424](https://doi.org/10.1126/science.1069424)
  * *Use for*: Block diagonal (#6), banded matrix (#7), LD structure analysis, chromosome-wise decomposition
- **Rao et al. (2014)** - 3D genome mapping - [DOI: 10.1016/j.cell.2014.11.021](https://doi.org/10.1016/j.cell.2014.11.021)
  * *Use for*: Sparse eigenvalue (#3), hierarchical (#10), Hi-C contact matrix analysis, chromatin compartments
- **GTEx Consortium (2020)** - Regulatory effects atlas - [DOI: 10.1126/science.aaz1776](https://doi.org/10.1126/science.aaz1776)
  * *Use for*: SVD (#2), NMF (#11), multi-tissue eQTL analysis, expression matrix factorization
