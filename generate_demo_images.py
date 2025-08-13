#!/usr/bin/env python3
"""
Generate demo images for README to showcase data collapse effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Set matplotlib style for better-looking plots
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.figsize'] = (10, 6)

def generate_demo_data():
    """Generate synthetic data that demonstrates data collapse."""
    np.random.seed(42)  # For reproducibility
    
    # System sizes
    L_values = [7, 9, 11, 13]
    
    # Control parameter range
    U_range = np.linspace(8.3, 9.0, 25)
    
    # True critical parameters
    U_c_true = 8.65
    a_true = 1.1
    
    # Generate data for each L
    all_data = []
    for L in L_values:
        # Scaling variable
        x_scaled = (U_range - U_c_true) * (L ** a_true)
        
        # Quadratic-like universal function
        Y_universal = 0.6 + 0.12 * x_scaled + 0.08 * (x_scaled**2)
        
        # Add finite-size corrections (multiplicative)
        Y_fse = Y_universal * (1 + 0.4 * L**(-0.8))
        
        # Add noise
        noise = 0.02 * np.random.randn(len(U_range))
        Y_noisy = Y_fse + noise
        
        # Store data
        for U, Y in zip(U_range, Y_noisy):
            all_data.append([L, U, Y])
    
    return np.array(all_data)

def plot_raw_data(data, save_path):
    """Plot raw data before collapse."""
    plt.figure(figsize=(10, 6))
    
    # Group by L and plot
    L_values = sorted(set(data[:, 0]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    for i, L in enumerate(L_values):
        mask = data[:, 0] == L
        U = data[mask, 1]
        Y = data[mask, 2]
        plt.plot(U, Y, 'o-', color=colors[i], linewidth=2, markersize=6, 
                label=f'L = {int(L)}', alpha=0.8)
    
    plt.xlabel(r'$U$ (Control Parameter)', fontsize=14)
    plt.ylabel(r'$R$ (Observable, dimensionless)', fontsize=14)
    plt.title('Raw Data Before Collapse', fontsize=16, fontweight='bold')
    plt.legend(title='System Size', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raw data plot to: {save_path}")

def plot_no_fse_collapse(data, save_path):
    """Plot data after collapse without finite-size correction."""
    plt.figure(figsize=(10, 6))
    
    U_c_true = 8.65
    a_true = 1.1
    
    L_values = sorted(set(data[:, 0]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    for i, L in enumerate(L_values):
        mask = data[:, 0] == L
        U = data[mask, 1]
        Y = data[mask, 2]
        x_scaled = (U - U_c_true) * (L ** a_true)
        plt.plot(x_scaled, Y, 'o-', color=colors[i], linewidth=2, markersize=6,
                label=f'L = {int(L)}', alpha=0.8)
    
    plt.xlabel(r'$(U - U_c) L^{1/\nu}$ (Scaling Variable)', fontsize=14)
    plt.ylabel(r'$R$ (Observable, dimensionless)', fontsize=14)
    plt.title('Data Collapse (without finite-size correction)', fontsize=16, fontweight='bold')
    plt.legend(title='System Size', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved No finite-size-correction plot to: {save_path}")

def plot_fse_collapse(data, save_path):
    """Plot data after collapse with finite-size correction."""
    plt.figure(figsize=(10, 6))
    
    U_c_true = 8.65
    a_true = 1.1
    b_true = 0.4
    c_true = -0.8
    
    L_values = sorted(set(data[:, 0]))
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    for i, L in enumerate(L_values):
        mask = data[:, 0] == L
        U = data[mask, 1]
        Y = data[mask, 2]
        x_scaled = (U - U_c_true) * (L ** a_true)
        Y_corrected = Y / (1 + b_true * (L ** c_true))
        plt.plot(x_scaled, Y_corrected, 'o-', color=colors[i], linewidth=2, markersize=6,
                label=f'L = {int(L)}', alpha=0.8)
    
    plt.xlabel(r'$(U - U_c) L^{1/\nu}$ (Scaling Variable)', fontsize=14)
    plt.ylabel(r'$R / (1 + b L^c)$ (Corrected Observable)', fontsize=14)
    plt.title('Data Collapse (with finite-size correction)', fontsize=16, fontweight='bold')
    plt.legend(title='System Size', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved with finite-size-correction plot to: {save_path}")

def main():
    print("Generating demo images for README...")
    data = generate_demo_data()
    images_dir = "docs/images"
    os.makedirs(images_dir, exist_ok=True)
    plot_raw_data(data, f"{images_dir}/raw_data.png")
    plot_no_fse_collapse(data, f"{images_dir}/nofse_collapse.png")
    plot_fse_collapse(data, f"{images_dir}/fse_collapse.png")
    print(f"\nAll demo images generated in: {images_dir}")

if __name__ == "__main__":
    main() 