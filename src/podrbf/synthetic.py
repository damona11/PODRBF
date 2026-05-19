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
    """
    Create a synthetic 3D time-resolved cylindrical dataset.

    Coordinates:
        x, y: cross-section coordinates
        z: axial coordinate
        time: time coordinate

    Variables:
        q: synthetic scalar field inside the cylinder
        mask: cylinder geometry mask
    """

    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.5, 1.5, ny)
    z = np.linspace(0, length, nz)
    time = np.linspace(0, 1, nt)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    mask = r <= radius

    q = np.zeros((nt, nx, ny, nz))

    for k, t in enumerate(time):
        axial_wave = np.sin(2 * np.pi * (Z / length - 2 * t))
        swirl_mode = np.cos(theta - 2 * np.pi * t)
        radial_profile = 1 - (r / radius) ** 2

        field = radial_profile * (1 + 0.3 * axial_wave + 0.2 * swirl_mode)
        field[~mask] = np.nan

        q[k] = field

    ds = xr.Dataset(
        data_vars={
            "q": (("time", "x", "y", "z"), q),
            "mask": (("x", "y", "z"), mask),
        },
        coords={
            "time": time,
            "x": x,
            "y": y,
            "z": z,
        },
        attrs={
            "case_name": "synthetic_cylinder",
            "geometry": "cylinder",
            "radius": radius,
            "length": length,
            "description": "Synthetic 3D time-resolved cylinder dataset for POD/RBF development.",
        },
    )

    ds["q"].attrs["units"] = "arbitrary"
    ds["q"].attrs["description"] = "Synthetic scalar field"

    return ds