#!/usr/bin/env bash
# Convenience launcher for the cuav_nmpc MPC.
# Sets the acados environment variables and runs a target script in the `mpc` conda env.
#
# Usage:
#   ./run.sh                # runs main.py (single-drone trajectory tracking)
#   ./run.sh main_cuav.py   # runs another script
#
# Requirements (already installed):
#   - acados built at ~/acados
#   - conda env `mpc` with acados_template, casadi, numpy, scipy, matplotlib, etc.

set -e

export ACADOS_SOURCE_DIR="$HOME/acados"
export DYLD_LIBRARY_PATH="$HOME/acados/lib:$DYLD_LIBRARY_PATH"

SCRIPT="${1:-main.py}"

# Use the mpc conda env's python directly.
exec "$HOME/miniconda3/envs/mpc/bin/python" "$SCRIPT"
