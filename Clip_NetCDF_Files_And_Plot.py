"""
Hybrid NetCDF Subreach Clipping Tool - Best of Both Scripts

Combines the robust coordinate validation and debugging from the first script
with the reach-specific CRS handling from the second script.

Features:
- Reads reach table to determine projection for each NetCDF file
- Handles State Plane coordinate systems (Illinois East/West)
- Maintains CSAT compatibility
- Enhanced validation and debugging
- Fallback to geographic coordinates if reach table lookup fails

Usage:
    python hybrid_clipper.py --nc_dir <path> --shp_file <path> --output_dir <path> --reach_table <path>
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
import pandas as pd
from pyproj import CRS
import logging
import netCDF4 as nc
from tqdm import tqdm


def setup_logging(output_dir):
    """Setup enhanced logging"""
    log_file = os.path.join(output_dir, 'hybrid_clipping.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_epsg_code(projection_name):
    """Convert text projection names to EPSG codes"""
    projection_dict = {
        'Illinois West': '3436',
        'Illinois East': '3435',
        'Illinois_West': '3436',
        'Illinois_East': '3435',
        'IL_West': '3436',
        'IL_East': '3435'
    }

    epsg = projection_dict.get(projection_name)
    if epsg is None:
        logging.warning(f"Unknown projection: {projection_name}")
        return None
    return epsg


def load_reach_table(reach_table_path):
    """Load the reach table with projection information"""
    logger = logging.getLogger()

    if not reach_table_path or not os.path.exists(reach_table_path):
        logger.warning("Reach table not found, will use fallback CRS detection")
        return None

    try:
        reach_table = pd.read_csv(reach_table_path)
        logger.info(f"Loaded reach table with {len(reach_table)} entries")

        # Print available columns and sample data
        logger.info(f"Reach table columns: {list(reach_table.columns)}")
        if len(reach_table) > 0:
            logger.info(f"Sample entry: {reach_table.iloc[0].to_dict()}")

        return reach_table
    except Exception as e:
        logger.error(f"Failed to load reach table: {e}")
        return None


def get_reach_crs(reach_name, reach_table):
    """Get CRS for a specific reach from the reach table"""
    logger = logging.getLogger()

    if reach_table is None:
        return None

    try:
        # Try different possible column names
        possible_id_cols = ['Reach_ID', 'reach_id', 'ReachID', 'reach_name', 'name']
        possible_proj_cols = ['Projection', 'projection', 'CRS', 'crs', 'EPSG']

        id_col = None
        proj_col = None

        for col in possible_id_cols:
            if col in reach_table.columns:
                id_col = col
                break

        for col in possible_proj_cols:
            if col in reach_table.columns:
                proj_col = col
                break

        if id_col is None or proj_col is None:
            logger.warning(f"Could not find required columns in reach table")
            logger.info(f"Available columns: {list(reach_table.columns)}")
            return None

        # Look up the reach
        matches = reach_table[reach_table[id_col] == reach_name]
        if len(matches) == 0:
            logger.warning(f"Reach {reach_name} not found in reach table")
            return None

        projection_name = matches[proj_col].iloc[0]
        epsg_code = get_epsg_code(projection_name)

        if epsg_code:
            logger.info(f"Found projection for {reach_name}: {projection_name} (EPSG:{epsg_code})")
            return CRS.from_epsg(int(epsg_code))
        else:
            return None

    except Exception as e:
        logger.error(f"Error looking up CRS for {reach_name}: {e}")
        return None


def detect_coordinate_system(ds, reach_name, reach_table):
    """Detect coordinate system using reach table first, then fallback methods"""
    logger = logging.getLogger()

    # Method 1: Try reach table lookup
    reach_crs = get_reach_crs(reach_name, reach_table)
    if reach_crs is not None:
        return reach_crs, "reach_table"

    # Method 2: Analyze coordinate ranges to infer system
    lats = ds['latitudes'].values
    lons = ds['longitudes'].values

    # Filter for finite values
    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    if valid_mask.sum() == 0:
        logger.error("No valid coordinates found")
        return None, "error"

    valid_lats = lats[valid_mask]
    valid_lons = lons[valid_mask]

    lat_range = valid_lats.max() - valid_lats.min()
    lon_range = valid_lons.max() - valid_lons.min()

    logger.info(f"Coordinate analysis for {reach_name}:")
    logger.info(f"  Latitude range: {valid_lats.min():.1f} to {valid_lats.max():.1f} (span: {lat_range:.1f})")
    logger.info(f"  Longitude range: {valid_lons.min():.1f} to {valid_lons.max():.1f} (span: {lon_range:.1f})")

    # Heuristics to detect coordinate system
    if (valid_lats.min() >= -90 and valid_lats.max() <= 90 and
            valid_lons.min() >= -180 and valid_lons.max() <= 360 and
            lat_range < 10 and lon_range < 10):  # Reasonable geographic ranges
        logger.info("Detected geographic coordinates (WGS84)")
        return CRS.from_epsg(4326), "geographic_detection"

    elif (abs(valid_lats.min()) > 100 or abs(valid_lons.min()) > 100):  # Large values suggest projected
        # Try to guess Illinois State Plane based on coordinate ranges
        if (valid_lons.min() > 300000 and valid_lons.max() < 500000 and
                valid_lats.min() > 300000 and valid_lats.max() < 500000):
            logger.info("Guessing Illinois East State Plane based on coordinate ranges")
            return CRS.from_epsg(3435), "range_detection"
        elif (valid_lons.min() > 200000 and valid_lons.max() < 400000 and
              valid_lats.min() > 300000 and valid_lats.max() < 500000):
            logger.info("Guessing Illinois West State Plane based on coordinate ranges")
            return CRS.from_epsg(3436), "range_detection"
        else:
            logger.warning("Detected projected coordinates but couldn't determine specific system")
            return CRS.from_epsg(3435), "default_state_plane"  # Default to Illinois East

    # Default fallback
    logger.warning("Could not determine coordinate system, defaulting to WGS84")
    return CRS.from_epsg(4326), "fallback"


def clip_netcdf_hybrid(nc_file, boundary, output_dir, reach_table, buffer_distance=0.0):
    """
    Clips NetCDF while preserving grid structure for CSAT compatibility

    CRITICAL: CSAT expects a regular 10ft x 10ft grid. We must preserve
    the grid structure and use fill values for points outside the boundary.
    """
    logger = logging.getLogger()

    try:
        reach_name = Path(nc_file).stem
        logger.info(f"\nProcessing: {reach_name}")

        # Open NetCDF file
        ds = xr.open_dataset(nc_file)

        # Detect coordinate system
        source_crs, detection_method = detect_coordinate_system(ds, reach_name, reach_table)
        logger.info(f"Using CRS: {source_crs} (detected via: {detection_method})")

        # Get ALL coordinates (don't filter by valid_range yet)
        lats = ds['latitudes'].values
        lons = ds['longitudes'].values

        logger.info(f"Original coordinate shapes - Lats: {lats.shape}, Lons: {lons.shape}")

        # Only filter out NaN/Inf in coordinates
        coord_valid_mask = np.isfinite(lats) & np.isfinite(lons)
        logger.info(f"Finite coordinates: {coord_valid_mask.sum()} / {len(lats)}")

        # Transform boundary to match NetCDF coordinate system
        logger.info(f"Transforming boundary from {boundary.crs} to {source_crs}")
        boundary_reproj = boundary.to_crs(source_crs)

        if len(boundary_reproj) > 1:
            boundary_reproj = boundary_reproj.dissolve()

        boundary_union = boundary_reproj.geometry.unary_union

        if buffer_distance > 0:
            logger.info(f"Applying {buffer_distance}m buffer to boundary...")
            boundary_buffered = boundary_union.buffer(buffer_distance)
        else:
            logger.info("Using exact boundary (no buffer)")
            boundary_buffered = boundary_union

        # Prepare geometry for faster operations
        from shapely import prepare
        prepare(boundary_buffered)

        logger.info("Performing spatial filtering...")

        # Create mask for points within boundary
        batch_size = 10000
        within_boundary_mask = np.zeros(len(lats), dtype=bool)

        # Only check points with valid coordinates
        valid_indices = np.where(coord_valid_mask)[0]

        for i in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[i:min(i + batch_size, len(valid_indices))]
            batch_lons = lons[batch_indices]
            batch_lats = lats[batch_indices]

            batch_points = [Point(lon, lat) for lon, lat in zip(batch_lons, batch_lats)]
            batch_mask = np.array([boundary_buffered.intersects(pt) for pt in batch_points])
            within_boundary_mask[batch_indices] = batch_mask

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_indices)}/{len(valid_indices)} points...")

        within_count = within_boundary_mask.sum()
        total_count = coord_valid_mask.sum()
        logger.info(f"Points within boundary: {within_count} / {total_count} ({within_count / total_count * 100:.1f}%)")

        if within_count == 0:
            logger.warning(f"No points from {reach_name} fall within boundary")
            extent = (lons[coord_valid_mask].min(), lats[coord_valid_mask].min(),
                      lons[coord_valid_mask].max(), lats[coord_valid_mask].max())
            create_no_overlap_plot(nc_file, boundary_reproj, extent, output_dir, source_crs)
            return None, extent

        # Get indices of points to keep
        clipped_indices = np.where(within_boundary_mask)[0]

        # Create clipped NetCDF file
        output_file = os.path.join(output_dir, f"{reach_name}.nc")

        with nc.Dataset(nc_file, 'r') as src, nc.Dataset(output_file, 'w') as dst:
            # Copy global attributes
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

            # Add clipping metadata
            dst.setncattr('clipping_applied', 'True')
            dst.setncattr('clipping_date', str(np.datetime64('now')))
            dst.setncattr('original_file', os.path.basename(nc_file))
            dst.setncattr('points_before_clipping', str(len(lats)))
            dst.setncattr('points_after_clipping', str(within_count))
            dst.setncattr('source_crs', str(source_crs))
            dst.setncattr('detection_method', detection_method)
            if buffer_distance > 0:
                dst.setncattr('buffer_applied', f'{buffer_distance}m')

            # Copy dimensions
            for name, dimension in src.dimensions.items():
                if name == 'points':
                    dst.createDimension(name, within_count)
                else:
                    dst.createDimension(name, len(dimension))

            # Copy variables preserving ALL attributes
            for name, variable in src.variables.items():
                try:
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)

                    # Copy ALL attributes (especially valid_range and _FillValue)
                    dst[name].setncatts({a: variable.getncattr(a) for a in variable.ncattrs()})

                    if 'points' in variable.dimensions:
                        if len(variable.dimensions) == 1:
                            # 1D variable (coordinates, etc.)
                            data = variable[:]
                            dst[name][:] = data[clipped_indices]
                        else:
                            # 2D variable (elevations: points x time)
                            data = variable[:]
                            # Clip along the points dimension
                            if variable.dimensions[0] == 'points':
                                dst[name][:] = data[clipped_indices, :]
                            else:
                                dst[name][:] = data[:, clipped_indices]
                    else:
                        # Copy non-spatial variables as-is
                        dst[name][:] = variable[:]

                except Exception as e:
                    logger.warning(f"Warning copying variable {name}: {e}")

        logger.info(f"Successfully clipped {reach_name}")
        extent = (lons[clipped_indices].min(), lats[clipped_indices].min(),
                  lons[clipped_indices].max(), lats[clipped_indices].max())

        ds.close()
        return output_file, extent

    except Exception as e:
        logger.error(f"Error clipping {nc_file}: {e}")
        if 'ds' in locals():
            ds.close()
        raise


def get_valid_coordinates(ds):
    """
    SIMPLIFIED: Only return coordinates, don't filter by valid_range

    Let the main function handle all filtering decisions
    """
    logger = logging.getLogger()

    lats = ds['latitudes'].values
    lons = ds['longitudes'].values

    logger.info(f"Coordinate shapes - Lats: {lats.shape}, Lons: {lons.shape}")

    # ONLY check for finite values
    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    logger.info(f"Finite coordinates: {valid_mask.sum()} / {len(lats)}")

    if valid_mask.sum() == 0:
        raise ValueError("No valid coordinates found")

    return lats[valid_mask], lons[valid_mask], valid_mask





def create_enhanced_plot(original_nc, clipped_nc, boundary_reproj, extent, output_dir, source_crs):
    """Create enhanced comparison plot with better zoom control"""
    logger = logging.getLogger()

    try:
        if clipped_nc is None:
            create_no_overlap_plot(original_nc, boundary_reproj, extent, output_dir, source_crs)
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Load datasets
        ds_orig = xr.open_dataset(original_nc)
        ds_clip = xr.open_dataset(clipped_nc)

        # Get valid coordinates for original
        orig_lats, orig_lons, orig_mask = get_valid_coordinates(ds_orig)
        clip_lats = ds_clip['latitudes'].values
        clip_lons = ds_clip['longitudes'].values

        # Plot 1: Original data
        ax1.scatter(orig_lons, orig_lats, c='blue', s=1, alpha=0.5, label='Original Points')
        boundary_reproj.boundary.plot(ax=ax1, color='red', linewidth=2, alpha=0.8)
        ax1.set_title(f'Original: {Path(original_nc).stem}\n({len(orig_lats)} points)')
        ax1.set_xlabel(f'X ({source_crs})')
        ax1.set_ylabel(f'Y ({source_crs})')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Clipped data
        ax2.scatter(clip_lons, clip_lats, c='green', s=1, alpha=0.7, label='Clipped Points')
        boundary_reproj.boundary.plot(ax=ax2, color='red', linewidth=2, alpha=0.8)
        ax2.set_title(f'Clipped: {Path(original_nc).stem}\n({len(clip_lats)} points)')
        ax2.set_xlabel(f'X ({source_crs})')
        ax2.set_ylabel(f'Y ({source_crs})')
        ax2.grid(True, alpha=0.3)

        # Optional: Set same x/y limits based on the boundary extent (with a margin)
        minx = clip_lons.min()
        maxx = clip_lons.max()
        miny = clip_lats.min()
        maxy = clip_lats.max()
        margin_x = (maxx - minx) * 0.1  # 10% padding
        margin_y = (maxy - miny) * 0.1

        xlim = (minx - margin_x, maxx + margin_x)
        ylim = (miny - margin_y, maxy + margin_y)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)

        # Add legend
        blue_patch = mpatches.Patch(color='blue', label='Original Points')
        green_patch = mpatches.Patch(color='green', label='Clipped Points')
        red_patch = mpatches.Patch(color='red', label='Boundary')
        fig.legend(handles=[blue_patch, green_patch, red_patch],
                   loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.tight_layout()

        # Save plot
        plot_name = f"{Path(original_nc).stem}_hybrid_comparison.png"
        plot_path = os.path.join(output_dir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plot saved: {plot_path}")

        ds_orig.close()
        ds_clip.close()

    except Exception as e:
        logger.error(f"Error creating enhanced plot: {e}")
        plt.close()



def create_no_overlap_plot(nc_file, boundary_reproj, extent, output_dir, source_crs):
    """Create diagnostic plot when no overlap found"""
    logger = logging.getLogger()

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ds_orig = xr.open_dataset(nc_file)
        orig_lats, orig_lons, _ = get_valid_coordinates(ds_orig)

        ax.scatter(orig_lons, orig_lats, c='blue', s=1, alpha=0.5, label='Survey Points')
        boundary_reproj.boundary.plot(ax=ax, color='red', linewidth=2, alpha=0.8)
        ax.set_title(f'No Overlap: {Path(nc_file).stem}\nCRS: {source_crs}')
        # Optional: Set same x/y limits based on the boundary extent (with a margin)
        minx = orig_lons.min()
        maxx = orig_lons.max()
        miny = orig_lats.min()
        maxy = orig_lats.max()
        margin_x = (maxx - minx) * 0.1  # 10% padding
        margin_y = (maxy - miny) * 0.1

        xlim = (minx - margin_x, maxx + margin_x)
        ylim = (miny - margin_y, maxy + margin_y)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax.set_xlabel(f'X ({source_crs})')
        ax.set_ylabel(f'Y ({source_crs})')
        ax.grid(True, alpha=0.3)

        # Add legend
        blue_patch = mpatches.Patch(color='blue', label='Survey Points')
        red_patch = mpatches.Patch(color='red', label='Boundary (No Overlap)')
        ax.legend(handles=[blue_patch, red_patch])

        plt.tight_layout()

        plot_name = f"{Path(nc_file).stem}_no_overlap.png"
        plot_path = os.path.join(output_dir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"No-overlap plot saved: {plot_path}")
        ds_orig.close()

    except Exception as e:
        logger.error(f"Error creating no-overlap plot: {e}")
        plt.close()


def process_files_hybrid(nc_dir, shp_file, output_dir, reach_table_path=None):
    """Main processing function combining both approaches"""

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("HYBRID NetCDF Clipping Tool")
    logger.info("=" * 50)
    logger.info(f"Input directory: {nc_dir}")
    logger.info(f"Shapefile: {shp_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reach table: {reach_table_path}")

    # Load boundary shapefile
    boundary = gpd.read_file(shp_file)
    logger.info(f"Boundary CRS: {boundary.crs}")
    logger.info(f"Boundary bounds: {boundary.total_bounds}")

    if len(boundary) > 1:
        logger.info("Multiple boundary features found, using union")
        boundary = boundary.dissolve()

    # Fix invalid geometries
    if not all(boundary.geometry.is_valid):
        logger.info("Fixing invalid geometries in boundary...")
        boundary.geometry = boundary.geometry.buffer(0)

    # Load reach table
    reach_table = load_reach_table(reach_table_path)

    # Find NetCDF files
    nc_files = list(Path(nc_dir).glob('*.nc'))
    logger.info(f"Found {len(nc_files)} NetCDF files")

    # Process files
    successful = 0
    no_overlap = 0
    failed = 0
    summary_data = []

    for nc_file in tqdm(nc_files, desc="Processing NetCDF files"):
        try:
            reach_name = nc_file.stem

            # Clip file
            clipped_file, extent = clip_netcdf_hybrid(str(nc_file), boundary, output_dir, reach_table)

            # Create plot
            if clipped_file:
                # Get CRS info for plotting
                with xr.open_dataset(clipped_file) as ds_clip:
                    source_crs, _ = detect_coordinate_system(xr.open_dataset(str(nc_file)), reach_name, reach_table)
                    boundary_reproj = boundary.to_crs(source_crs)
                    create_enhanced_plot(str(nc_file), clipped_file, boundary_reproj, extent, output_dir, source_crs)

                successful += 1

                # Add to summary
                with xr.open_dataset(str(nc_file)) as ds_orig, xr.open_dataset(clipped_file) as ds_clip:
                    total_points = len(ds_orig['latitudes'])
                    clipped_points = len(ds_clip['latitudes'])
                    summary_data.append({
                        'Reach_Name': reach_name,
                        'Total_Points': total_points,
                        'Points_Kept': clipped_points,
                        'Percent_Kept': (clipped_points / total_points * 100),
                        'CRS_Used': str(source_crs)
                    })
            else:
                no_overlap += 1
                summary_data.append({
                    'Reach_Name': reach_name,
                    'Total_Points': len(xr.open_dataset(str(nc_file))['latitudes']),
                    'Points_Kept': 0,
                    'Percent_Kept': 0,
                    'CRS_Used': 'No overlap'
                })

        except Exception as e:
            logger.error(f"Failed to process {nc_file.name}: {e}")
            failed += 1

    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'hybrid_clipping_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved: {summary_file}")

    # Copy reach table if it exists
    if reach_table_path and os.path.exists(reach_table_path):
        try:
            import shutil
            shutil.copy2(reach_table_path, output_dir)
            logger.info("Copied reach table to output directory")
        except Exception as e:
            logger.warning(f"Failed to copy reach table: {e}")

    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("PROCESSING SUMMARY:")
    logger.info(f"Successfully clipped: {successful} files")
    logger.info(f"No overlap found: {no_overlap} files")
    logger.info(f"Failed: {failed} files")
    logger.info(f"Total files: {len(nc_files)}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid NetCDF clipping with reach-specific CRS handling')
    parser.add_argument('--nc_dir', required=True, help='Directory containing .nc files')
    parser.add_argument('--shp_file', required=True, help='Path to boundary shapefile')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--reach_table', help='Path to reach table with CRS info (optional)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.nc_dir):
        print(f"Error: NetCDF directory does not exist: {args.nc_dir}")
        sys.exit(1)

    if not os.path.isfile(args.shp_file):
        print(f"Error: Shapefile does not exist: {args.shp_file}")
        sys.exit(1)

    try:
        process_files_hybrid(args.nc_dir, args.shp_file, args.output_dir, args.reach_table)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_with_cemvr_paths():
    """Run with CEMVR paths"""
    nc_folder = "C:/workspace/CSAT/CSAT_distribution/data/CEMVR"
    shapefile_path = "C:/workspace/CSAT/CSAT_distribution/data/testshp/MVR_AOI.shp"
    output_folder = "C:/workspace/CSAT/CSAT_distribution/data/Clipped"
    reach_table_path = "C:/workspace/CSAT/CSAT_distribution/data/CEMVR/reach_table.txt"

    try:
        process_files_hybrid(nc_folder, shapefile_path, output_folder, reach_table_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("Running hybrid clipper with CEMVR paths...")
        success = run_with_cemvr_paths()
        if success:
            print("Hybrid clipping completed successfully!")
        else:
            print("Clipping failed. Check the log for details.")