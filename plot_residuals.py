import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot residuals',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--matrix', type=str, default='advection', choices=['advection', 'possion'],
                        help='Matrix to use for the problem')
    parser.add_argument('--N', type=int, default=50,
                        help='Grid size')
    parser.add_argument('--nu', type=int, default=30,
                        help='Number of smoothing iterations')
    return parser.parse_args()
# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main import problem1

args = parse_args()
# Run the same parameters as your command
results = problem1(
    matrix=args.matrix,
    N_values=[args.N],  
    num_levels=3,
    nu=args.nu,
    smooth_params=0.667,
    smoother='jacobi',
    tol=1e-8,
    max_iterations=200,
    use_galerkin=True,  # Use Galerkin coarsening
    solver_type='multigrid',
    test_mode=True,
)

res = results[0]
residuals = res['rel_residuals']
iterations = range(len(residuals))

# Detect convergence phase by tracking reduction factor (ρ)
# Convergence phase: ρ is consistently small (good reduction)
# Stagnation phase: ρ ≈ 1.0 (no more reduction, just noise)
convergence_phase = len(residuals) - 1
if len(residuals) > 10:
    rhos = [residuals[i]/residuals[i-1] if residuals[i-1] > 0 else 1.0 
            for i in range(1, len(residuals))]
    
    # Use a sliding window to detect when ρ stops being consistently small
    window_size = 5
    rho_threshold = 0.9  # If average ρ > 0.95, we're in stagnation
    
    for i in range(window_size, len(rhos)):
        window_avg_rho = np.prod(rhos[i-window_size:i]) ** (1/(window_size+1))
        if window_avg_rho > rho_threshold and residuals[i] < 1e-8:
            convergence_phase = i - window_size
            break
    
    # Also compute avg ρ during convergence phase only
    if convergence_phase > 1:
        convergence_rhos = rhos[:convergence_phase]
        # Filter out very small values that might cause issues with log
        convergence_rhos = [r for r in convergence_rhos if r > 1e-15]
        if len(convergence_rhos) > 0:
            avg_rho_convergence = np.exp(np.mean(np.log(convergence_rhos)))
        else:
            avg_rho_convergence = 0
    else:
        avg_rho_convergence = 0
else:
    avg_rho_convergence = 0

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(iterations, residuals, 'b-o', linewidth=2, markersize=3, label=f'N={args.N}', alpha=0.7)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Relative Residual ||r||/||b|| (log scale)', fontsize=12)
ax.set_title(f'Multigrid Convergence ({args.matrix}, N={args.N}, ν={args.nu}, smoother=jacobi)', fontsize=14)
ax.grid(True, alpha=0.3, which='both')

# Mark convergence phase boundary if it exists
if convergence_phase < len(residuals):
    ax.axvline(x=convergence_phase, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.5, label=f'Convergence achieved (iter {convergence_phase})')
    ax.legend(fontsize=10)

# Add info box
info_text = f'Total iterations: {len(residuals)-1}\nConverged at: iteration {convergence_phase}\nFinal residual: {residuals[-1]:.2e}'
ax.text(0.98, 0.95, info_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('residual_history.png', dpi=150, bbox_inches='tight')
print('\n' + '='*60)
print('✓ Plot saved to: residual_history.png')
print('='*60)
plt.show()

