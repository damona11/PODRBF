# scripts/visualize_cylinder.py

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt

import xarray as xr 
from podrbf.data_handling.data_visualization import (
    plot_crosssection,
    plot_axial_slice,
    plot_time_evolution,
    plot_velocity_magnitude,
    plot_summary,
)

def main():
    ds = xr.open_zarr(str(Path(__file__).resolve().parents[1] / "data" / "synthetic" / "synthetic_cylinder.zarr"))
    plot_crosssection(ds, field='u', t_idx=0)
    plot_crosssection(ds, field='v', t_idx=0)
    plot_crosssection(ds, field='w', t_idx=0)
    plot_axial_slice(ds, field='u', t_idx=0)
    plot_axial_slice(ds, field='w', t_idx=0)
    plot_time_evolution(ds, field='u')
    plot_time_evolution(ds, field='v')
    plot_time_evolution(ds, field='w')
    plot_velocity_magnitude(ds, t_idx=0)
    plot_summary(ds, t_idx=0)
    plt.show()

if __name__ == "__main__":
    main()



