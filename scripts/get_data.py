"""Download the KU Leuven BEV energy-dynamics dataset (CC BY 4.0, DOI 10.48804/8KPDTW) and unpack the driving sessions.

Yasko, M., Moussa, A., Tian, F., Kazmi, H., Driesen, J., Martinez, W. (2025). Unveiling Energy Dynamics of Battery
Electric Vehicles Using High-Resolution Data. Scientific Data. Data: https://doi.org/10.48804/8KPDTW
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

DOI = "doi:10.48804/8KPDTW"
BASE = "https://rdr.kuleuven.be"
ZIP_NAME = "BEV energy dynamic data_V2.zip"


VED_BASE = "https://raw.githubusercontent.com/gsoh/VED/master/Data"
VED_PARTS = ["VED_DynamicData_Part1.7z", "VED_DynamicData_Part2.7z"]


def get_ved(dest: str = "data/raw/ved") -> int:
    """Vehicle Energy Dataset (Apache-2.0): dynamic data for the external-validation Nissan Leafs. Needs `pip install py7zr`."""
    import py7zr

    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    for name in VED_PARTS:
        target = dest_p / name
        if not target.exists():
            with requests.get(f"{VED_BASE}/{name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(target, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        with py7zr.SevenZipFile(target) as z:
            z.extractall(dest_p)
    print("ok:", dest_p)
    return 0


def main(dest: str = "data/raw") -> int:
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    meta = requests.get(f"{BASE}/api/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60)
    meta.raise_for_status()
    files = meta.json()["data"]["latestVersion"]["files"]
    target = next((f for f in files if f["label"] == ZIP_NAME), None)
    if target is None:
        print(f"{ZIP_NAME} not found in dataset version; available: {[f['label'] for f in files]}", file=sys.stderr)
        return 1
    zpath = dest_p / "bev.zip"
    if not zpath.exists() or zpath.stat().st_size != target["dataFile"]["filesize"]:
        with requests.get(f"{BASE}/api/access/datafile/{target['dataFile']['id']}", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zpath, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    if zpath.stat().st_size != target["dataFile"]["filesize"]:
        print("download size mismatch", file=sys.stderr)
        return 1
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if "driving sessions/" in n or "Additional files/" in n:
                z.extract(n, dest_p)
    print("ok:", dest_p)
    return 0


if __name__ == "__main__":
    if "--ved" in sys.argv:
        raise SystemExit(get_ved())
    raise SystemExit(main(*[a for a in sys.argv[1:] if not a.startswith("--")]))
