import numpy as np
import matplotlib.pyplot as plt

def decompose_into_fourier_modes(error, n):
    """
    Decompose error into Fourier sine modes: w_{k,j} = sin(j*k*π/n)
    
    This is the standard discrete Fourier decomposition for grid functions
    on [0,1] with zero boundary conditions.
    
    Args:
        error: Error vector at interior grid points (length n-1)
        n: Number of grid intervals (N in problem, where n = N-1 interior points)
    
    Returns:
        coefficients: Fourier coefficients α_k for k=1,...,n-1
        modes: Matrix where modes[k-1, j] = w_{k,j} = sin(j*k*π/n)
    """
    n_interior = len(error)  # This is n-1 in the notation
    N = n_interior + 1       # Number of grid intervals
    
    # Create the sine mode basis
    # modes[k-1, j] = sin(j*k*π/N) for k=1,...,n_interior and j=1,...,n_interior
    modes = np.zeros((n_interior, n_interior))
    for k in range(1, n_interior + 1):
        for j in range(1, n_interior + 1):
            modes[k-1, j-1] = np.sin(j * k * np.pi / N)
    
    # Compute Fourier coefficients using orthogonality
    # α_k = (2/N) * sum_{j=1}^{n-1} error[j] * sin(j*k*π/N)
    coefficients = np.zeros(n_interior)
    for k in range(1, n_interior + 1):
        coefficients[k-1] = (2.0 / N) * np.sum(error * modes[k-1, :])
    
    return coefficients, modes


def decompose_error_frequencies(error, n):
    """
    Decompose error into high and low frequency components using FFT
    
    Args:
        error: Error vector (spatial domain)
        n: Size of the vector
    
    Returns:
        high_freq: High frequency component
        low_freq: Low frequency component
    """
    # Use FFT to decompose into frequency components
    error_fft = np.fft.fft(error)
    
    # Split at the midpoint frequency
    mid_freq = n // 2
    
    # Create masks for low and high frequencies
    low_freq_fft = error_fft.copy()
    high_freq_fft = error_fft.copy()
    
    # Zero out high frequencies for low-frequency component
    low_freq_fft[mid_freq:n-mid_freq+1] = 0
    
    # Zero out low frequencies for high-frequency component
    high_freq_fft[:mid_freq] = 0
    high_freq_fft[n-mid_freq+1:] = 0
    
    # Transform back to spatial domain
    low_freq = np.real(np.fft.ifft(low_freq_fft))
    high_freq = np.real(np.fft.ifft(high_freq_fft))
    
    return high_freq, low_freq


def plot_fourier_mode_analysis(results, save_path='fourier_modes.png', max_modes=None):
    """
    Plot Fourier mode coefficient decay during multigrid iterations
    
    Args:
        results: Results from problem1()
        save_path: Path to save the figure
        max_modes: Maximum number of modes to display (default: all modes)
    """
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 5*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, res in enumerate(results):
        N = res['N']
        u_true = res['u_true']
        solution_history = res['solution_history']
        n_interior = len(u_true)
        
        # Determine number of modes to plot
        n_modes = max_modes if max_modes is not None else n_interior
        n_modes = min(n_modes, n_interior)
        
        # Compute Fourier coefficients at each iteration
        all_coefficients = []
        for x in solution_history:
            error = x - u_true
            coeffs, _ = decompose_into_fourier_modes(error, n_interior)
            all_coefficients.append(np.abs(coeffs))  # Store absolute values
        
        all_coefficients = np.array(all_coefficients)  # Shape: (num_iters, n_interior)
        
        # Plot 1: Mode amplitudes over iterations (heatmap)
        ax1 = axes[idx, 0]
        mode_indices = np.arange(1, n_modes + 1)
        iterations = np.arange(len(solution_history))
        
        # Create heatmap
        im = ax1.imshow(all_coefficients[:, :n_modes].T, 
                       aspect='auto', cmap='hot', 
                       norm=plt.matplotlib.colors.LogNorm(vmin=1e-16, vmax=1.0),
                       origin='lower', interpolation='nearest')
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Mode k', fontsize=12)
        ax1.set_title(f'N={N}: Fourier Mode Amplitudes |α_k(iter)|', 
                     fontsize=13, fontweight='bold')
        ax1.set_yticks(np.arange(0, n_modes, max(1, n_modes//10)))
        ax1.set_yticklabels(np.arange(1, n_modes+1, max(1, n_modes//10)))
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('|α_k|', fontsize=11)
        
        # Add contour lines to show convergence
        ax1.axhline(y=n_interior//2, color='cyan', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Mid frequency')
        ax1.legend(fontsize=9, loc='upper right')
        
        # Plot 2: Selected mode trajectories
        ax2 = axes[idx, 1]
        
        # Select representative modes: low, mid, high frequency
        selected_modes = []
        labels = []
        colors = []
        
        # Low frequency modes (k=1,2,3)
        for k in [1, 2, 3]:
            if k <= n_interior:
                selected_modes.append(k-1)
                labels.append(f'k={k} (low)')
                colors.append('blue')
        
        # Mid frequency modes
        mid_k = n_interior // 2
        for offset in [-1, 0, 1]:
            k = mid_k + offset
            if 1 <= k <= n_interior:
                selected_modes.append(k-1)
                labels.append(f'k={k} (mid)')
                colors.append('green')
        
        # High frequency modes
        for k in [n_interior-2, n_interior-1, n_interior]:
            if k >= 1:
                selected_modes.append(k-1)
                labels.append(f'k={k} (high)')
                colors.append('red')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_modes = []
        unique_labels = []
        unique_colors = []
        for m, l, c in zip(selected_modes, labels, colors):
            if m not in seen:
                seen.add(m)
                unique_modes.append(m)
                unique_labels.append(l)
                unique_colors.append(c)
        
        # Plot selected modes
        for mode_idx, label, color in zip(unique_modes, unique_labels, unique_colors):
            ax2.semilogy(iterations, all_coefficients[:, mode_idx], 
                        marker='o', linewidth=2, markersize=4, 
                        label=label, color=color, alpha=0.8)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Mode Amplitude |α_k|', fontsize=12)
        ax2.set_title(f'N={N}: Selected Fourier Mode Convergence', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([1e-16, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFourier mode analysis plot saved to: {save_path}")
    
    return fig


def plot_frequency_decomposition(results, save_path='frequency_decomposition.png'):
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5*len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, res in enumerate(results):
        N = res['N']
        u_true = res['u_true']
        solution_history = res['solution_history']
        x_grid = res['x_grid']
        n = len(u_true)
        
        # Compute error decomposition at each iteration
        high_freq_norms = []
        low_freq_norms = []
        total_error_norms = []
        
        for x in solution_history:
            error = x - u_true
            high_freq, low_freq = decompose_error_frequencies(error, n)
            
            high_freq_norms.append(np.linalg.norm(high_freq))
            low_freq_norms.append(np.linalg.norm(low_freq))
            total_error_norms.append(np.linalg.norm(error))
        
        # Plot 1: Error decay over iterations (log scale)
        ax1 = axes[idx, 0]
        iterations = np.arange(len(solution_history))
        ax1.semilogy(iterations, total_error_norms, 'k-o', label='Total Error', linewidth=2, markersize=5)
        ax1.semilogy(iterations, high_freq_norms, 'r--s', label='High Freq Error', linewidth=2, markersize=5)
        ax1.semilogy(iterations, low_freq_norms, 'b--^', label='Low Freq Error', linewidth=2, markersize=5)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('L2 Norm of Error', fontsize=12)
        ax1.set_title(f'N={N}: Error Decay by Frequency', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frequency composition over iterations (stacked)
        ax2 = axes[idx, 1]
        high_ratio = np.array(high_freq_norms) / np.array(total_error_norms)
        low_ratio = np.array(low_freq_norms) / np.array(total_error_norms)
        
        ax2.plot(iterations, high_ratio, 'r-o', label='High Freq %', linewidth=2, markersize=5)
        ax2.plot(iterations, low_ratio, 'b-s', label='Low Freq %', linewidth=2, markersize=5)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Ratio to Total Error', fontsize=12)
        ax2.set_title(f'N={N}: Error Composition', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # Plot 3: Spatial error at selected iterations
        ax3 = axes[idx, 2]
        # Show initial, middle, and final errors
        selected_iters = [0, len(solution_history)//2, len(solution_history)-1]
        colors = ['r', 'orange', 'g']
        
        for i, iter_idx in enumerate(selected_iters):
            if iter_idx < len(solution_history):
                error = solution_history[iter_idx] - u_true
                ax3.plot(x_grid, error, color=colors[i], linewidth=2, 
                        label=f'Iter {iter_idx} (||e||={total_error_norms[iter_idx]:.2e})', alpha=0.8)
        
        ax3.set_xlabel('x', fontsize=12)
        ax3.set_ylabel('Error', fontsize=12)
        ax3.set_title(f'N={N}: Spatial Error Evolution', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFrequency decomposition plot saved to: {save_path}")
    
    return fig