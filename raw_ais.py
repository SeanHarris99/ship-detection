import numpy as np
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
import logging


def build(date: str) -> pd.DataFrame:

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
