# src/podrbf/data_handling/create_cylinder_data.py

import numpy as np
import xarray as xr


def create_cylinder_dataset(
    nx=80,
    ny=80,
    nz=60,
    nt=100,
    radius=1.0,
    length=4.0,
):
    """conda list -n podrbf | findstr -i "qt matplotlib pyqt"
    Create a synthetic 3D time-resolved cylindrical dataset.

    Coordinates:
        x, y : cross-section coordinates
        z    : axial coordinate
        time : time coordinate

    Variables:
        u    : x-velocity component  (nx, ny, nz, nt)
        v    : y-velocity component  (nx, ny, nz, nt)
        w    : z-velocity component  (nx, ny, nz, nt)
        mask : cylinder geometry mask (nx, ny, nz)
    """

    x    = np.linspace(-1.5, 1.5, nx)
    y    = np.linspace(-1.5, 1.5, ny)
    z    = np.linspace(0, length, nz)
    time = np.linspace(0, 1, nt)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r     = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    mask  = r <= radius

    # radial profile — parabolic, zero at wall
    radial_profile = 1 - (r / radius) ** 2
    radial_profile[~mask] = 0.0

    u = np.zeros((nx, ny, nz, nt))
    v = np.zeros((nx, ny, nz, nt))
    w = np.zeros((nx, ny, nz, nt))

    for k, t in enumerate(time):

        # axial wave travelling in z
        axial_wave   = np.sin(2 * np.pi * (Z / length - 2 * t))

        # swirl mode rotating in time
        swirl_mode   = np.cos(theta - 2 * np.pi * t)

        # u — x-velocity: swirl driven
        u[..., k]  = radial_profile * (
              0.3 * np.cos(theta - 2 * np.pi * t)
            + 0.1 * axial_wave
        )

        # v — y-velocity: swirl driven, 90 deg out of phase with u
        v[..., k]  = radial_profile * (
              0.3 * np.sin(theta - 2 * np.pi * t)
            + 0.1 * axial_wave
        )

        # w — z-velocity: axial plug flow + axial wave
        w[..., k]  = radial_profile * (
              1.0
            + 0.2 * axial_wave
            + 0.1 * swirl_mode
        )

        # apply mask — NaN outside cylinder
        u[~mask, k] = np.nan
        v[~mask, k] = np.nan
        w[~mask, k] = np.nan

    ds = xr.Dataset(
        data_vars={
            "u"    : (("x", "y", "z", "time"), u),
            "v"    : (("x", "y", "z", "time"), v),
            "w"    : (("x", "y", "z", "time"), w),
            "mask" : (("x", "y", "z"),          mask),
        },
        coords={
            "x"    : x,
            "y"    : y,
            "z"    : z,
            "time" : time,
        },
        attrs={
            "case_name"   : "synthetic_cylinder",
            "geometry"    : "cylinder",
            "radius"      : radius,
            "length"      : length,
            "description" : "Synthetic 3D time-resolved cylinder dataset for POD/RBF development.",
        },
    )

    ds["u"].attrs = {"units": "m/s", "description": "x-velocity component"}
    ds["v"].attrs = {"units": "m/s", "description": "y-velocity component"}
    ds["w"].attrs = {"units": "m/s", "description": "z-velocity component"}

    return ds