"""
SPDX-License-Identifier:    GPL-3.0-only
Copyright (C) 2026 Hippolyte Audet-Lagacé
Author:         Hippolyte Audet-Lagacé
Description:    This program takes as input a locations.bin file of little-endians byte ordering [f32, 3] xyz unit 
                sphere cartesian coordinate values and generates an elevations.bin file of the same shape containing
                elevation at the location (+-10m at worse precision), population standard deviation and relief.
                Alternatively, if you pulled this from my (author) original repo, I should have at least included the 
                completed elevations.bin file for the 9-subdivision QSC I'm using in dev. Prints progress to console
                every 5% completion increment.
                This code can probably easily be adapted to read from any TIF file.
Example call :  python -u elevation.py locations.bin elevations.bin
For testing :   python -u elevation.py test_locations.bin test_elevations.bin
"""

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from datetime import datetime
import rasterio
from rasterio.windows import Window

# Constants -----------------------------------------------------------------------------------------------------------
 
GRID_STEP_DEG: float = 1.0 / 60.0   # ETOPO1 pixel spacing in degrees
DEFAULT_DATASET_PATH: str = r'path\to\your\tif\ETOPO1.tif'      # TODO change this part if you want to use this script
DEFAULT_DATASET: str = "ETOPO1.tif"
DEFAULT_GRID_N: int = 7              # default NxN (odd N) grid size, 7x7 datapoints for an ~11x11km square region

# I/O helpers ---------------------------------------------------------------------------------------------------------

def read_positions(filename: str) -> np.ndarray:
    """
    Reads a .bin file expecting [f32, 3] triplets of values in little-endians byte ordering.
    Returns a np.ndarray of shape (M, 3), dtype float32
    """
    size = os.path.getsize(filename)
    if size % 12 != 0:
        raise ValueError(f"File size {size} is not divisible by 12 bytes per vertex")
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape((-1, 3))

def write_measures(filename: str, measures: np.ndarray) -> None:
    """
    Writes an (M, 3) float32 array of [elevation, std, relief] triplets to a
    .bin file using little-endian byte ordering.
    filename : destination path
    measures : np.ndarray of shape (M, 3), dtype float32
    """
    if measures.ndim != 2 or measures.shape[1] != 3:
        raise ValueError(f"measures must have shape (M, 3), got {measures.shape}")
    # Warning if precision loss (non f32 arrays are passed)
    if measures.dtype != np.float32 and np.issubdtype(measures.dtype, np.floating):
        import warnings
        warnings.warn(
            f"measures has dtype {measures.dtype}; values will be truncated to float32.",
            RuntimeWarning,
            stacklevel=2,
        )
    out = measures.astype("<f4")    # little-endian f32 ordering
    out.tofile(filename)

# Data manipulation ---------------------------------------------------------------------------------------------------

def cartesian_to_latlon(xyz: np.ndarray) -> np.ndarray:
    """
    Converts unit-sphere Cartesian coordinates to (lat, lon) in degrees (EPSG:4326 / WGS-84 convention).
    xyz : np.ndarray, shape (M, 3), (x, y, z) on the unit sphere
    Returns a np.ndarray, shape (M, 2), (latitude in [-90, 90], longitude in [-180, 180])
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.stack([lat_deg, lon_deg], axis=1)

def interpolate_elevation_bicubic(
    lats_grid: np.ndarray,
    lons_grid: np.ndarray,
    elevation_grid: np.ndarray,
    lat: float,
    lon: float,
) -> float:
    """
    Estimate elevation at (lat, lon) via bicubic spline interpolation over an NxN grid (N >= 4)
    Uses scipy.interpolate.RectBivariateSpline with k=3 (cubic)
    lats_grid      : np.ndarray of latitude  values (ascending)
    lons_grid      : np.ndarray of longitude values (ascending)
    elevation_grid : np.ndarray of elevations, shape (n, n)
    lat, lon       : query point in degrees
    Returns the interpolated elevation in metres, dtype float
    """
    if elevation_grid.shape[0] < 4 or elevation_grid.shape[1] < 4:
        raise ValueError("Grid must be at least 4 by 4 for cubic interpolation")
    # RectBivariateSpline expects (x=lat, y=lon) in ascending order
    spline = RectBivariateSpline(lats_grid, lons_grid, elevation_grid, kx=3, ky=3)
    return float(spline(lat, lon)[0, 0])

def compute_std(elevation_grid: np.ndarray) -> float:
    """
    Standard deviation of all finite elevation values in the grid.
    """
    valid = elevation_grid[np.isfinite(elevation_grid)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid))
 
 
def compute_relief(elevation_grid: np.ndarray) -> float:
    """
    Relief (max elevation - min elevation) over all finite grid values.
    """
    valid = elevation_grid[np.isfinite(elevation_grid)]
    if valid.size == 0:
        return float("nan")
    return float(np.max(valid) - np.min(valid))

# Dataset readsy ------------------------------------------------------------------------------------------------------

def snap_to_grid(lat: float, lon: float) -> tuple[float, float]:
    """
    Snaps (lat, lon) to the nearest ETOPO1 grid node (multiples of 1 arc-minute).
    """
    lat_snapped = round(lat / GRID_STEP_DEG) * GRID_STEP_DEG
    lon_snapped = round(lon / GRID_STEP_DEG) * GRID_STEP_DEG
    return lat_snapped, lon_snapped

def get_elevation_grid_rasterio(
    lat: float,
    lon: float,
    n: int = DEFAULT_GRID_N,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an NxN elevation grid centered on the nearest grid node to (lat, lon) directly from a GeoTIFF.
    """
    if abs(abs(lat) - 90.0) < GRID_STEP_DEG:
        pole_lat = 90.0 if lat > 0 else -90.0
        lats_grid = pole_lat + np.arange(-(n // 2), n // 2 + 1) * GRID_STEP_DEG
        lats_grid = np.clip(lats_grid, -90.0, 90.0)
        lons_grid = np.arange(-(n // 2), n // 2 + 1) * GRID_STEP_DEG
        with rasterio.open(dataset_path) as src:
            row, col = src.index(0.0, pole_lat)
            row = int(np.clip(row, 0, src.height - 1))
            col = int(np.clip(col, 0, src.width - 1))
            window = Window(col, row, 1, 1)
            pole_elevation = float(src.read(1, window=window)[0, 0])
        elevation_grid = np.full((n, n), pole_elevation, dtype=np.float64)
        return lats_grid, lons_grid, elevation_grid

    centre_lat, centre_lon = snap_to_grid(lat, lon)
    half = n // 2
    lats_grid = centre_lat + np.arange(-half, half + 1) * GRID_STEP_DEG
    lons_grid = centre_lon + np.arange(-half, half + 1) * GRID_STEP_DEG

    with rasterio.open(dataset_path) as src:
        # Read the entire NxN window in one call
        row, col = src.index(centre_lon, centre_lat)
        row = int(np.clip(row - half, 0, src.height - n))
        col = int(np.clip(col - half, 0, src.width - n))
        window = Window(col, row, n, n)
        elevation_grid = src.read(1, window=window).astype(np.float64)

    return lats_grid, lons_grid, elevation_grid

# Main and it's buddy -------------------------------------------------------------------------------------------------

def process_positions(
    positions: np.ndarray,
    n: int = DEFAULT_GRID_N,
    dataset_path: str = DEFAULT_DATASET_PATH,
    verbose: bool = True,
) -> np.ndarray:
    """
    For each unit-sphere XYZ position, compute (elevation, std, relief) by:
      1. Converting to (lat, lon) in degrees (EPSG:4326 / WGS-84 convention)
      2. Reading an NxN raw-elevation grid directly from the GeoTIFF
      3. Bicubic-interpolating to get the elevation at the exact location
      4. Computing standard deviation and relief over the grid
    positions    : np.ndarray, shape (M, 3)
    n            : NxN grid size
    dataset_path : path to the ETOPO1 GeoTIFF file
    verbose      : print progress
    Returns an np.ndarray, shape (M, 3), dtype float32, [elevation_m, std_m, relief_m]
    """
    latlon = cartesian_to_latlon(positions)
    m = len(positions)
    measures = np.full((m, 3), fill_value=np.nan, dtype=np.float32)

    for i, (lat, lon) in enumerate(latlon):
        try:
            lats_g, lons_g, elev_g = get_elevation_grid_rasterio(lat, lon, n=n, dataset_path=dataset_path)
            relief = compute_relief(elev_g)
            if relief == 0.0:
                elevation = float(elev_g[n // 2, n // 2])
            else:
                elevation = interpolate_elevation_bicubic(lats_g, lons_g, elev_g, lat, lon)
            std = compute_std(elev_g)
            measures[i] = (elevation, std, relief)
        except Exception as exc:
            print(f"  WARNING: point {i} ({lat:.4f}, {lon:.4f}) failed: {exc}")

        if verbose and (i % max(1, m // 20) == 0 or i == m - 1):
            print(f"  [{i + 1}/{m}] {datetime.now().strftime('%H:%M:%S')}")

    return measures

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract ETOPO1 elevation/std/relief for unit-sphere .bin positions."
    )
    parser.add_argument("input",  help="Input .bin file  ([f32,3] XYZ unit-sphere)")
    parser.add_argument("output", help="Output .bin file ([f32,3] elevation/std/relief)")
    parser.add_argument(
        "--grid-n", type=int, default=DEFAULT_GRID_N, metavar="N",
        help=f"NxN grid size (odd, >= 5; default {DEFAULT_GRID_N})"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, metavar="NAME",
        help=f"Dataset name (default {DEFAULT_DATASET})"
    )
    args = parser.parse_args()
 
    if args.grid_n % 2 == 0 or args.grid_n < 3:
        parser.error("--grid-n must be an odd integer >= 5")
 
    print(f"Reading positions from  : {args.input}")
    positions = read_positions(args.input)
    print(f"  {len(positions)} points loaded")
    print(f"Grid size               : {args.grid_n}x{args.grid_n}")
    print(f"Dataset                 : {args.dataset}")

    print("Processing start")
    measures = process_positions(
        positions,
        n=args.grid_n,
    )
 
    print("Writing start")
    write_measures(args.output, measures)
    print(f"Wrote {len(measures)} triplets to {args.output}")
 
    # This should indicate any catastrophic failure, as reference: everest height < 10000m, challenger deep > -11000m
    elev = measures[:, 0]
    std  = measures[:, 1]
    rel  = measures[:, 2]
    print(f"  elevation : min={np.nanmin(elev):.1f}  max={np.nanmax(elev):.1f}  mean={np.nanmean(elev):.1f} m")
    print(f"  std       : min={np.nanmin(std):.1f}   max={np.nanmax(std):.1f}   mean={np.nanmean(std):.1f} m")
    print(f"  relief    : min={np.nanmin(rel):.1f}   max={np.nanmax(rel):.1f}   mean={np.nanmean(rel):.1f} m")
 
if __name__ == "__main__":
    main()