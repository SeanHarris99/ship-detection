import numpy as np
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
import logging

from pystac_client import Client
from shapely.geometry import shape, Point
import geopandas as gpd


def build_ais(date: str) -> pd.DataFrame:

    # pull coast guard data
    # -------------------------------------------------------------------------
    target_date = pd.to_datetime(date).date()
    fname = f"AIS_2023_{target_date.month:02d}_{target_date.day:02d}.zip"
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/{fname}"

    r = requests.get(url, timeout=60)
    with ZipFile(BytesIO(r.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            ais = pd.read_csv(f)

    # column renames
    # -------------------------------------------------------------------------
    column_renames = columns = {
        "MMSI": "mmsi",
        "BaseDateTime": "utc",
        "LAT": "lat",
        "LON": "lon",
        "SOG": "sog",
        "COG": "cog",
        "Heading": "heading",
        "VesselName": "name",
        "IMO": "imo",
        "CallSign": "call_sign",
        "VesselType": "vessel_type",
        "Status": "status",
        "Length": "length",
        "Width": "width",
        "Draft": "draft",
        "Cargo": "cargo",
        "TransceiverClass": "transceiver_class",
    }

    ais = ais.rename(columns=column_renames)

    # filters
    # -------------------------------------------------------------------------
    # large cargo ships, not personal vessels
    ais = ais[ais.transceiver_class == "A"]

    # drops ships in harbor, bad data
    ais = ais[(ais.sog > 1) & (ais.sog < 80)]

    # drops small ships, bad data
    ais = ais[(ais.length > 30) & (ais.length < 400)]

    # heading 511 is nonrespondor, set to nan
    ais = ais.replace({"heading": {511: np.nan}})

    # set index
    # -------------------------------------------------------------------------
    ais = (
        ais.sort_values(by=["mmsi", "utc"])
        .assign(ping=lambda x: x.groupby("mmsi").cumcount())
        .set_index(["mmsi", "ping"])
    )

    # typing
    # -------------------------------------------------------------------------
    ais["utc"] = pd.to_datetime(ais["utc"], utc=True)

    return ais


def build_tiles(date: str) -> gpd.GeoDataFrame:
    # sat_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    sat_url = "https://earth-search.aws.element84.com/v1"
    sat_api = Client.open(sat_url)

    results = sat_api.search(
        collections=["sentinel-1-grd"],
        bbox=[-160, 4, -50, 50],
        datetime=date,
        limit=200,
        max_items=None,
    )

    sat_tiles = []
    for item in results.get_all_items():
        idx = item.id
        geom = shape(item.geometry)
        dt = item.datetime
        sat_tiles.append((idx, geom, dt))

    gdf = gpd.GeoDataFrame(
        sat_tiles,
        columns=["id", "geometry", "date"],
        geometry="geometry",
        crs="EPSG:4326",
    )

    gdf = gdf.set_index("id")

    return gdf


def build_chips(ais: pd.DataFrame, sat_tiles: gpd.GeoDataFrame) -> pd.DataFrame:

    ais = gpd.GeoDataFrame(
        ais, geometry=gpd.points_from_xy(ais.lon, ais.lat), crs="EPSG:4326"
    )

    buffer = pd.Timedelta(minutes=5)
    sat_tiles["buffer_start_time"] = sat_tiles["date"] - buffer
    sat_tiles["buffer_end_time"] = sat_tiles["date"] + buffer

    intersection = gpd.sjoin(ais, sat_tiles, predicate="within")

    intersection = intersection[
        (intersection["utc"] >= intersection["buffer_start_time"])
        & (intersection["utc"] <= intersection["buffer_end_time"])
    ]

    intersection = intersection.rename(
        columns={
            "utc": "start_time",
            "date": "pred_time",
            # "lat": "start_lat",
            # "lon": "start_lon"
        }
    )

    KNOTS_IN_KM = 1.852
    SECONDS_IN_HOUR = 3600
    KM_PER_DEGREE = 111.32

    intersection["time_delta"] = (
        intersection["pred_time"] - intersection["start_time"]
    ).dt.total_seconds()
    intersection["travelled"] = (intersection["sog"] * KNOTS_IN_KM) * (
        intersection["time_delta"] / SECONDS_IN_HOUR
    )

    intersection["pred_lat"] = (
        intersection["lat"]
        + (intersection["travelled"] * np.cos(np.radians(intersection["cog"])))
        / KM_PER_DEGREE
    )
    intersection["pred_lon"] = intersection["lon"] + (
        intersection["travelled"] * np.sin(np.radians(intersection["cog"]))
    ) / (KM_PER_DEGREE * np.cos(np.radians(intersection["lat"])))

    return intersection
