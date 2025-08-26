import numpy as np
import matplotlib.pyplot as plt

# Generate two 1D sine waves
t = np.linspace(0, 4*np.pi, 1000)  # time vector
sine1 = np.sin(t)                  # first sine wave
sine2 = np.sin(2*t + np.pi/4)      # second sine wave with different frequency and phase

# Generate two 1D cosine waves (similar to sine waves)
x = np.linspace(0, 4*np.pi, 1000)      # same length as sine waves
cosine1 = np.cos(x)                    # first cosine wave
cosine2 = np.cos(2*x + np.pi/4)        # second cosine wave with different frequency and phase

# Compute outer product
outer_product_cos = np.outer(cosine1, cosine2)
outer_product_sin = np.outer(sine1, sine2)

print(f"Sine wave 1 shape: {sine1.shape}")
print(f"Sine wave 2 shape: {sine2.shape}")
print(f"Outer product shape: {outer_product_sin.shape}")

print(f"Cosine wave 1 shape: {cosine1.shape}")
print(f"Cosine wave 2 shape: {cosine2.shape}")
print(f"Outer product shape: {outer_product_cos.shape}")

# visualize both sine waves
plt.plot(t, sine1, label='Sine wave 1')
plt.plot(t, sine2, label='Sine wave 2')
plt.legend()
plt.show()

# visualize both cosine waves
plt.plot(x, cosine1, label='Cosine wave 1')
plt.plot(x, cosine2, label='Cosine wave 2')
plt.legend()
plt.show()


# visualize outer product
plt.imshow(outer_product_sin, cmap='viridis')
plt.colorbar()
plt.show()

plt.imshow(outer_product_cos, cmap='viridis')
plt.colorbar()
plt.show()

# sum of outer products
sum_of_outer_products = outer_product_sin + outer_product_cos
plt.imshow(sum_of_outer_products, cmap='viridis')
plt.colorbar()
plt.show()

# svd on sum of outer products
U, S, V = np.linalg.svd(sum_of_outer_products)

# Golden ratio for figure dimensions
golden_ratio = 1.618

# Create golden ratio heatmaps for U, V, and Sigma
plt.figure(figsize=(15, 15/golden_ratio))

# U matrix heatmap
plt.subplot(1, 3, 1)
plt.imshow(U, cmap='viridis', aspect='auto')
plt.title('U Matrix')
plt.colorbar()

# Sigma diagonal matrix heatmap
plt.subplot(1, 3, 2)
sigma_matrix = np.diag(S)
plt.imshow(sigma_matrix, cmap='plasma', aspect='auto')
plt.title('Sigma Matrix')
plt.colorbar()

# V matrix heatmap
plt.subplot(1, 3, 3)
plt.imshow(V, cmap='magma', aspect='auto')
plt.title('V Matrix')
plt.colorbar()

plt.tight_layout()
plt.show()
