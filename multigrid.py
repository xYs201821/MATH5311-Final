"""
Multigrid Solver Implementation for 1D Problems
"""

import numpy as np
from tqdm import tqdm


class Multigrid:
    """
    Multigrid hierarchy for restriction and interpolation operators
    """
    def __init__(self, n_finest, num_levels):
        self.n_finest = n_finest
        self.num_levels = num_levels
        self.R_hierarchy = []
        self.I_hierarchy = []
        self.n_hierarchy = []
        
        # Finest level
        n = n_finest
        self.n_hierarchy.append(n)
        
        for level in range(num_levels):
            if level < num_levels - 1:
                R, n_coarse = self._create_restriction_matrix(n)
                
                self.R_hierarchy.append(R)
                self.I_hierarchy.append(2 * R.T)
                
                n = n_coarse
                self.n_hierarchy.append(n)
    
    def restrict(self, v_fine, level):
        """Restrict fine grid vector to coarse grid"""
        return self.R_hierarchy[level] @ v_fine
    
    def interpolate(self, v_coarse, level):
        """Interpolate coarse grid vector to fine grid"""
        return self.I_hierarchy[level] @ v_coarse
    
    def get_grid_size(self, level):
        """Get the grid size at a given level"""
        return self.n_hierarchy[level]

    def _create_restriction_matrix(self, n):
        """Create full-weighting restriction matrix"""
        n_coarse = n // 2
        R = np.zeros((n_coarse, n))
        for i in range(n_coarse):
            col_center = 2 * i + 1
            R[i, col_center - 1] = 0.25
            R[i, col_center] = 0.5
            if col_center + 1 < n:
                R[i, col_center + 1] = 0.25
        return R, n_coarse


def jacobi(A, b, x, omega=2/3, num_iterations=3, tol=1e-20, smooth=True):
    """
    Weighted Jacobi smoother
    
    Args:
        A: System matrix
        b: Right-hand side
        x: Current solution
        omega: Damping parameter 
        num_iterations: Number of smoothing iterations
    
    Returns:
        x: Smoothed solution
    """
    n = len(b)
    D = np.diag(A)
    for _ in range(num_iterations):
        x_new = x.copy()
        r = b - A @ x
        if not smooth and np.linalg.norm(r) < tol:
            break
        x = x_new + omega * r / D
    return x


def gauss_seidel(A, b, x, num_iterations=3, omega=1.0):
    """
    Gauss-Seidel smoother
    
    Args:
        A: System matrix
        b: Right-hand side
        x: Current solution
        num_iterations: Number of smoothing iterations
        omega: Damping parameter (for consistency, typically not used)
    
    Returns:
        x: Smoothed solution
    """
    n = len(b)
    
    for _ in range(num_iterations):
        for i in range(n):
            sigma = A[i, :i].dot(x[:i]) + A[i, i+1:].dot(x[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]
    return x


def multigrid_vcycle(Multigrid, A, b, x, level, smoother='jacobi', 
                    nu=2, coarsest_level=2, smooth_params=2/3, 
                    matrix_constructor=None, use_galerkin=True):
    """
    Recursive multigrid V-cycle
    
    Args:
        Multigrid: Multigrid object with hierarchy
        A: System matrix at current level
        b: Right-hand side at current level
        x: Current solution
        level: Current grid level
        smoother: Smoother type ('jacobi' or 'gauss_seidel')
        nu: Number of pre/post smoothing iterations
        coarsest_level: Level at which to use direct solve
        smooth_params: Damping parameter for smoother
        matrix_constructor: Function to construct matrix at each level (for non-symmetric operators)
        use_galerkin: If False, use matrix_constructor directly; if True, use R·A·I
    
    Returns:
        x: Updated solution after V-cycle
    """
    if level == Multigrid.num_levels - 1 or level == coarsest_level - 1:
        x = np.linalg.solve(A, b) # exact solver
        #x = jacobi(A, b, x, num_iterations=10000, omega=1.0, smooth=False)
        return x

    # Step 1: Pre-smoothing
    if smoother == 'jacobi':
        x = jacobi(A, b, x, num_iterations=nu, omega=smooth_params)
    else:
        x = gauss_seidel(A, b, x, num_iterations=nu, omega=smooth_params)
    
    r = b - A @ x
    R = Multigrid.R_hierarchy[level]
    I = Multigrid.I_hierarchy[level]
    # Step 2: restrict residual to coarse grid
    r_coarse = R @ r
    
    # Step 3: solve error on coarse grid
    # Construct coarse-grid operator
    if use_galerkin:
        # Use Galerkin coarsening (works for symmetric operators)
        A_coarse = R @ A @ I
    else:
        n_coarse = Multigrid.get_grid_size(level + 1)
        A_coarse = matrix_constructor(n_coarse)
    e_coarse = np.zeros(len(r_coarse))
    e_coarse = multigrid_vcycle(Multigrid, A_coarse, r_coarse, e_coarse, 
                                level + 1, smoother, nu, coarsest_level, smooth_params,
                                matrix_constructor, use_galerkin)
    # Step 4: interpolate error back to fine grid
    e_fine = I @ e_coarse
    
    x = x + e_fine
    # Step 5: post-smoothing
    
    if smoother == 'jacobi':
        x = jacobi(A, b, x, num_iterations=nu, omega=smooth_params)
    else:
        x = gauss_seidel(A, b, x, num_iterations=nu, omega=smooth_params)
    
    return x


class JacobiSolver:
    """
    Simple Jacobi iterative solver for comparison with multigrid
    """
    
    def __init__(self, omega=2/3, max_iterations=1e6, tol=1e-8):
        """
        Initialize Jacobi solver
        
        Args:
            omega: Damping parameter
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.omega = omega
        self.max_iterations = max_iterations
        self.tol = tol
        self.residual_history = []
        self.solution_history = []
    
    def solve(self, A, b, x0=None):
        """
        Solve the linear system Ax = b using Jacobi iteration
        
        Args:
            A: System matrix
            b: Right-hand side
            x0: Initial guess (default: zero vector)
        
        Returns:
            x: Solution
            info: Dictionary with convergence information
        """
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        self.residual_history = []
        self.solution_history = []
        
        D = np.diag(A)
        
        r = b - A @ x
        initial_residual_norm = np.linalg.norm(r)
        self.residual_history.append(initial_residual_norm / np.linalg.norm(b))
        self.solution_history.append(x.copy())
        
        pbar = tqdm(desc=f"Jacobi (ω={self.omega:.3f})", unit="iter", 
                   leave=True, position=0, dynamic_ncols=True, mininterval=0.1)
        
        # Jacobi iterations
        for iteration in range(int(self.max_iterations)):
            # One Jacobi iteration
            r = b - A @ x
            x = x + self.omega * r / D
            
            # Compute residual
            r = b - A @ x
            residual_norm = np.linalg.norm(r) / np.linalg.norm(b)
            self.residual_history.append(residual_norm)
            self.solution_history.append(x.copy())
            
            # Calculate convergence factor
            rho = residual_norm / self.residual_history[-2] if len(self.residual_history) > 1 else 1.0
            
            # Update progress bar
            pbar.set_postfix({
                'residual': f'{residual_norm:.2e}',
                'rho': f'{rho:.4f}'
            })
            pbar.update(1)
            
            # Check convergence
            if residual_norm < self.tol:
                pbar.close()
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'residual_norm': residual_norm * np.linalg.norm(b),
                    'relative_residual': residual_norm,
                    'residual_history': self.residual_history
                }
                return x, info
        
        pbar.close()
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'residual_norm': residual_norm * np.linalg.norm(b),
            'relative_residual': residual_norm,
            'residual_history': self.residual_history
        }
        return x, info
    
    def get_residual_history(self):
        """Get the residual history"""
        return self.residual_history


class MultigridSolver:
    """
    High-level multigrid solver with iteration tracking
    """
    
    def __init__(self, num_levels=2, coarsest_level=2, smoother='jacobi', 
                 nu=2, smooth_params=2/3, max_iterations=1e6, tol=1e-8,
                 matrix_constructor=None, use_galerkin=True, test_mode=False):
        """
        Initialize multigrid solver
        
        Args:
            num_levels: Number of multigrid levels
            coarsest_level: Level at which to use direct solve
            smoother: Smoother type ('jacobi' or 'gauss_seidel')
            nu: Number of pre/post smoothing iterations
            smooth_params: Damping parameter for smoother
            max_iterations: Maximum number of V-cycle iterations
            tol: Convergence tolerance
            matrix_constructor: Function to construct operator at each level (for non-symmetric ops)
            use_galerkin: Use Galerkin coarsening (True) or direct construction (False)
        """
        self.num_levels = num_levels
        self.coarsest_level = coarsest_level
        self.smoother = smoother
        self.nu = nu
        self.smooth_params = smooth_params
        self.max_iterations = max_iterations
        self.tol = tol
        self.matrix_constructor = matrix_constructor
        self.use_galerkin = use_galerkin
        self.multigrid = None
        self.residual_history = []
        self.solution_history = []  # Track solutions at each iteration
    
    def solve(self, A, b, x0=None, test_mode=False):
        """
        Solve the linear system Ax = b using multigrid V-cycle
        
        Args:
            A: System matrix
            b: Right-hand side
            x0: Initial guess (default: zero vector)
        
        Returns:
            x: Solution
            info: Dictionary with convergence information
        """
        n = len(b)
        
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        self.multigrid = Multigrid(n_finest=n, num_levels=self.num_levels)
        
        self.residual_history = []
        self.solution_history = []
        
        r = b - A @ x
        initial_residual_norm = np.linalg.norm(r)
        self.residual_history.append(initial_residual_norm)
        self.solution_history.append(x.copy())
        
        pbar = tqdm(desc=f"N={n+1} Multigrid", unit="iter")
        
        # V-cycle iterations
        for iteration in range(int(self.max_iterations)):
            # Perform one V-cycle
            x = multigrid_vcycle(self.multigrid, A, b, x, level=0, 
                               smoother=self.smoother, nu=self.nu, 
                               coarsest_level=self.coarsest_level, smooth_params=self.smooth_params,
                               matrix_constructor=self.matrix_constructor, use_galerkin=self.use_galerkin)
            
            # Compute residual
            r = b - A @ x
            residual_norm = np.linalg.norm(r)/np.linalg.norm(b)
            self.residual_history.append(residual_norm)
            self.solution_history.append(x.copy())
            
            # Calculate one-step decrease factor
            rho = residual_norm / self.residual_history[-2] if len(self.residual_history) > 1 else 1.0
            
            # Update progress bar
            pbar.set_postfix({
                'residual': f'{residual_norm/np.linalg.norm(b):.2e}',
                'rho': f'{rho:.4f}'
            })
            pbar.update(1)
            # Check convergence
            if residual_norm < self.tol * np.linalg.norm(b) and test_mode==False:
                break
        # Check convergence
        if residual_norm < self.tol * np.linalg.norm(b):
            pbar.close()
            info = {
                'converged': True,
                'iterations': iteration + 1,
                'residual_norm': residual_norm,
                'relative_residual': residual_norm / np.linalg.norm(b),
                'residual_history': self.residual_history,
                'solution_history': self.solution_history
            }
            return x, info
        
        pbar.close()
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'residual_norm': residual_norm,
            'relative_residual': residual_norm / np.linalg.norm(b),
            'residual_history': self.residual_history,
            'solution_history': self.solution_history
        }
        return x, info
    
    def get_residual_history(self):
        """Get the residual history"""
        return self.residual_history

