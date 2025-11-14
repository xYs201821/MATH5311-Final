# MATH5311 Final-Project

A Python implementation of the multigrid V-cycle method for solving 1D Poisson equation and advection equation. 

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Basic Examples

```bash
# Run multigrid solver with default settings
python main.py

# Solve Poisson equation with 2-steps jacobi smoother
python main.py 

# Compare with Jacobi-only solver
python main.py --solver jacobi-only -m 500

# Solve advection problem with 1-step gauss-seidel smoother and 3-level V-cycle
python main.py --matrix advection -l 3 -s gauss_seidel -n 1

# Plot residual history for different smoothing iterations
python plot_residuals.py --matrix advection --N 50 --nu 30
python plot_residuals.py --matrix advection --N 50 --nu 50
```

## Command-Line Arguments

### main.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-N`, `--grid-sizes` | `[50, 100, 200]` | Grid sizes to test |
| `-l`, `--num-levels` | `2` | Number of multigrid levels |
| `-n`, `--nu` | `2` | Pre/post smoothing iterations |
| `-w`, `--omega` | `0.667` | Damping parameter (ω) |
| `-s`, `--smoother` | `jacobi` | Smoother type (`jacobi`, `gauss_seidel`) |
| `-t`, `--tol` | `1e-8` | Convergence tolerance |
| `-m`, `--max-iter` | `100000` | Maximum iterations |
| `--matrix` | `possion` | Matrix type (`possion`, `advection`) |
| `--solver` | `multigrid` | Solver type (`multigrid`, `jacobi-only`) |
| `--no-galerkin` | `False` | Use direct construction instead of Galerkin |
| `--plot-freq` | `False` | Generate frequency decomposition plot |
| `--plot-modes` | `False` | Generate Fourier mode analysis |
| `--test-mode` | `False` | Continue iterations even after convergence |

### plot_residuals.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--matrix` | `advection` | Matrix type (`advection`, `possion`) |
| `--N` | `50` | Grid size |
| `--nu` | `30` | Number of smoothing iterations |

## Files

- `main.py` - Main driver program with CLI and configuration printing
- `multigrid.py` - Multigrid and Jacobi solver implementations
- `plot_residuals.py` - Residual history plotting with convergence phase detection
- `requirements.txt` - Python dependencies


## Acknowledgement

This project was completed as part of final project for MATH5311 in HKUST. The implementation draws from online resources including *A Multigrid Tutorial* and the MIT course on multigrid delivered by Prof. Gilbert Strang. Code comments and this README file were formulated with the help of Cursor.