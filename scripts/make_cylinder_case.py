from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from podrbf.synthetic import create_cylinder_dataset


def main():
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = create_cylinder_dataset(
        nx=80,
        ny=80,
        nz=60,
        nt=100,
        radius=1.0,
        length=4.0,
    )

    output_file = output_dir / "synthetic_cylinder.zarr"

    ds.to_zarr(output_file, mode="w")

    print(ds)
    print(f"Saved dataset to: {output_file}")


if __name__ == "__main__":
    main()

