"""
MATH5311 - Final Project
"""

import warnings
import argparse
import numpy as np
from multigrid import MultigridSolver, JacobiSolver

from plot import plot_frequency_decomposition, plot_fourier_mode_analysis

# Suppress overflow/divide warnings that clutter output
warnings.filterwarnings('ignore', category=RuntimeWarning)
def create_1d_possion_matrix(n):
    """
    Create 1D Poisson matrix for n interior points
    
    Args:
        n: Number of interior points
    
    Returns:
        A: n x n Poisson matrix
    """
    h = 1.0 / (n + 1)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0 / h**2
        if i > 0:
            A[i, i-1] = -1.0 / h**2
        if i < n-1:
            A[i, i+1] = -1.0 / h**2
    return A

def create_1d_advection_matrix(n):
    """
    Create 1D advection matrix for n interior points
    
    Args:
        n: Number of interior points
    
    Returns:
        A: n x n advection matrix
    """
    h = 1.0 / (n + 1)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1.0 / h
        if i > 0:
            A[i, i-1] = -1.0 / h
    return A

def problem1(matrix='poisson', N_values=[50, 100, 200], num_levels=2, nu=2, smooth_params=2/3, 
             smoother='jacobi', tol=1e-8, max_iterations=1e6, use_galerkin=True, solver_type='multigrid'):
    """
    Solve 1D Poisson equation using multigrid V-cycle or Jacobi iteration
    Args:
        matrix: Matrix to use for the problem ('poisson' or 'advection')
        N_values: List of grid sizes (including boundaries)
        num_levels: Number of multigrid levels
        nu: Number of smoothing iterations (pre and post)
        smooth_params: Damping parameter (omega) for smoother
        smoother: Smoother type ('jacobi' or 'gauss_seidel')
        tol: Convergence tolerance
        max_iterations: Maximum iterations
        use_galerkin: Use Galerkin coarsening (True) or direct construction (False)
        solver_type: Type of solver ('multigrid' or 'jacobi-only')
    """
    if matrix == 'possion':
        create_matrix = create_1d_possion_matrix
    elif matrix == 'advection':
        create_matrix = create_1d_advection_matrix
    else:
        raise ValueError(f"Invalid matrix: {matrix}")
    results = []
    for N in N_values:  
        n = N - 1 # number of interior points
        x_grid = np.linspace(1.0/N, 1.0-1.0/N, N-1)
        A = create_matrix(n)
        b = np.ones(n)
        x0 = np.ones(n) # initial guess
        if matrix == 'possion':
            u_true = x_grid * (1.0 - x_grid) / 2
        elif matrix == 'advection':
            u_true = x_grid
        else:
            raise ValueError(f"Invalid matrix: {matrix}")
        
        # Choose solver based on solver_type
        if solver_type == 'jacobi-only':
            solver = JacobiSolver(omega=smooth_params, max_iterations=max_iterations, tol=tol)
        else:  # multigrid
            solver = MultigridSolver(num_levels=num_levels, smoother=smoother, nu=nu, 
                                    coarsest_level=num_levels, smooth_params=smooth_params,
                                    tol=tol, max_iterations=max_iterations,
                                    matrix_constructor=create_matrix, use_galerkin=use_galerkin)

        x, info = solver.solve(A, b, x0)
        
        # Extract history from solver
        rel_residuals = info['residual_history']
        
        # Calculate convergence factors
        rhos = [1.0]
        for i in range(1, len(rel_residuals)):
            rho = rel_residuals[i] / rel_residuals[i-1]
            rhos.append(rho)
        # Calculate final errors (only the final solution matters for plotting)
        errors_sup = [np.linalg.norm(x - u_true, np.inf)]
        errors_l2 = [np.linalg.norm(x - u_true)]
        
        results.append({
            'N': N,
            'solution': x,
            'u_true': u_true,
            'x_grid': x_grid,
            'solution_history': solver.solution_history,
            'errors_sup': errors_sup,
            'errors_l2': errors_l2,
            'rel_residuals': rel_residuals,
            'rhos': rhos,
            'info': info
        })
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multigrid solver for 1D Poisson equation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-N', '--grid-sizes', type=int, nargs='+', 
                        default=[50, 100, 200],
                        help='Grid sizes (including boundaries) to test')
    
    parser.add_argument('-l', '--num-levels', type=int, default=2,
                        help='Number of multigrid levels')
    
    parser.add_argument('-n', '--nu', type=int, default=2,
                        help='Number of pre/post smoothing iterations')
    
    parser.add_argument('-w', '--omega', type=float, default=2/3,
                        dest='smooth_params',
                        help='Damping parameter (omega) for smoother')
    
    parser.add_argument('-s', '--smoother', type=str, default='jacobi',
                        choices=['jacobi', 'gauss_seidel'],
                        help='Smoother type')
    
    parser.add_argument('-t', '--tol', type=float, default=1e-8,
                        help='Convergence tolerance')
    
    parser.add_argument('-m', '--max-iter', type=int, default=int(1e6),
                        dest='max_iterations',
                        help='Maximum number of iterations')
    
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    parser.add_argument('--plot-freq', action='store_true',
                        help='Plot frequency decomposition of errors (FFT-based)')
    
    parser.add_argument('--plot-modes', action='store_true',
                        help='Plot Fourier mode analysis (sine basis decomposition)')
    
    parser.add_argument('-o', '--output', type=str, default='frequency_decomposition.png',
                        help='Output filename for frequency decomposition plot')
    
    parser.add_argument('--modes-output', type=str, default='fourier_modes.png',
                        help='Output filename for Fourier mode analysis plot')
    
    parser.add_argument('--max-modes', type=int, default=None,
                        help='Maximum number of Fourier modes to display')
    parser.add_argument('--matrix', type=str, default='possion', choices=['possion', 'advection'],
                        help='Matrix to use for the problem')
    parser.add_argument('--no-galerkin', action='store_false', dest='use_galerkin',
                        help='Use Galerkin coarsening (True) or direct construction (False)')
    parser.add_argument('--solver', type=str, default='multigrid', 
                        choices=['multigrid', 'jacobi-only'],
                        help='Solver type: multigrid V-cycle or Jacobi-only iteration')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run solver
    results = problem1(
        matrix=args.matrix,
        N_values=args.grid_sizes,
        num_levels=args.num_levels,
        nu=args.nu,
        smooth_params=args.smooth_params,
        smoother=args.smoother,
        tol=args.tol,
        max_iterations=args.max_iterations,
        use_galerkin=args.use_galerkin,
        solver_type=args.solver
    )
    
    print("\n" + "="*60)
    print("MULTIGRID CONVERGENCE SUMMARY")
    print("="*60)
    for res in results:
        N = res['N']
        iters = res['info']['iterations']
        avg_rho = np.mean(res['rhos'][1:]) if len(res['rhos']) > 1 else 0
        final_error = res['errors_sup'][0]
        print(f"N={N:3d}: {iters:3d} iters, avg_rho={avg_rho:.4f}, error={final_error:.2e}")
    print("="*60)
    
    # Plot frequency decomposition if requested
    if args.plot_freq:
        plot_frequency_decomposition(results, save_path=args.output)
    
    # Plot Fourier mode analysis if requested
    if args.plot_modes:
        plot_fourier_mode_analysis(results, save_path=args.modes_output, 
                                   max_modes=args.max_modes)


