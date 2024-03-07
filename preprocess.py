#preprocess.py
#preprocess.py
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from pathlib import Path
import numpy as np


def preprocess_rasters(dmsp_dir, bm_dir, selected_country,
                       shapefile_path="World_Countries/World_Countries_Generalized.shp"):
    # Load the country shapefile
    countries = gpd.read_file(shapefile_path)
    country_shape = countries[countries['COUNTRY'] == selected_country]

    if country_shape.empty:
        raise ValueError(f"Country '{selected_country}' not found in the shapefile.")

    # List all DMSP raster files
    dmsp_raster_paths = list(Path(dmsp_dir).glob('*.tif'))

    # Assuming you want to pair each DMSP raster with the first BM raster found
    bm_raster_path = next(Path(bm_dir).glob('*.tif'), None)
    if not bm_raster_path:
        raise FileNotFoundError("No BM raster files found in the specified directory.")

    for dmsp_raster_path in dmsp_raster_paths:
        with rasterio.open(dmsp_raster_path) as dmsp_src, rasterio.open(bm_raster_path) as bm_src:
            # Print input resolutions
            print(f"Input DMSP resolution: {dmsp_src.res}")
            print(f"Input BM resolution: {bm_src.res}")

            # Reproject BM raster to match DMSP raster's resolution
            bm_resampled = rasterio.io.MemoryFile()
            with bm_resampled.open(**dmsp_src.meta) as bm_dst:
                reproject(
                    source=rasterio.band(bm_src, 1),
                    destination=rasterio.band(bm_dst, 1),
                    src_transform=bm_src.transform,
                    src_crs=bm_src.crs,
                    dst_transform=dmsp_src.transform,
                    dst_crs=dmsp_src.crs,
                    resampling=Resampling.bilinear
                )

                # Print output resolution (should match DMSP resolution)
                print(f"Output BM resolution (matching DMSP): {bm_dst.res}")

                # Crop the BM raster
                bm_out_image, _ = mask(bm_dst, country_shape.geometry, crop=True)

                # Set values above 5000 as 0
                bm_out_image = np.where(bm_out_image > 5000, 0, bm_out_image)

                bm_cropped_path = Path(
                    bm_dir) / f"cropped_{bm_raster_path.stem}_{dmsp_raster_path.stem}{bm_raster_path.suffix}"
                with rasterio.open(bm_cropped_path, "w", **bm_dst.meta) as bm_out:
                    bm_out.write(bm_out_image)

            # Crop the DMSP raster without changing resolution and print output resolution
            dmsp_out_image, _ = mask(dmsp_src, country_shape.geometry, crop=True)
            print(f"Output DMSP resolution (unchanged): {dmsp_src.res}")

            dmsp_cropped_path = Path(dmsp_dir) / f"cropped_{dmsp_raster_path.name}"
            with rasterio.open(dmsp_cropped_path, "w", **dmsp_src.meta) as dmsp_out:
                dmsp_out.write(dmsp_out_image)

            print(f"DMSP raster cropped and saved to {dmsp_cropped_path}")
            print(f"BM raster cropped and saved to {bm_cropped_path}")