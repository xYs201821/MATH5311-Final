# MATH5311 Final-Project

A Python implementation of the multigrid V-cycle method for solving 1D possion equation and advection equation.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Basic Examples

```bash
# Run multigrid solver with default settings
python main.py

# Solve possion equation with 2-steps jacobi smoother
python main.py 

# Compare with Jacobi-only solver
python main.py --solver jacobi-only -m 500

# Solve advection problem with 1-step gauss-seidel smoother and 3-level V-cycle
python main.py --matrix advection -l 3 -s gauss_seidel -n 1

```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-N`, `--grid-sizes` | `[50, 100, 200]` | Grid sizes to test |
| `-l`, `--num-levels` | `2` | Number of multigrid levels |
| `-n`, `--nu` | `2` | Pre/post smoothing iterations |
| `-w`, `--smooth-params` | `0.667` | Damping parameter (ω) |
| `--smoother` | `jacobi` | Smoother type (`jacobi`, `gauss_seidel`) |
| `--tol` | `1e-8` | Convergence tolerance |
| `-m`, `--max-iterations` | `1000000` | Maximum iterations |
| `--matrix` | `laplacian` | Matrix type (`laplacian`, `advection`) |
| `--solver` | `multigrid` | Solver type (`multigrid`, `jacobi-only`) |
| `--no-galerkin` | `False` | Use direct construction instead of Galerkin |
| `--plot-freq` | `False` | Generate frequency decomposition plot |
| `--plot-modes` | `False` | Generate Fourier mode analysis |

## Example Results

### Multigrid vs Jacobi (N=50)

| Solver | Iterations | Avg ρ | Convergence |
|--------|-----------|-------|-------------|
| **Multigrid** | 6 | 0.036 | ✓ Fast |
| **Jacobi-only** | 200+ | 0.998 | ✗ Slow |


## Files

- `main.py` - Main driver program with CLI
- `multigrid.py` - Multigrid and Jacobi solver implementations
- `plot.py` - Visualization and analysis functions
- `requirements.txt` - Python dependencies


