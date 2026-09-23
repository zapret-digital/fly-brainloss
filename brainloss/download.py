"""Download the connectome tables (checked by size and MD5).

MaleCNS v1.0 (CC BY 4.0), about 0.57 GB:
    https://male-cns.janelia.org/download/
FlyWire FAFB v783, about 0.88 GB, only needed for the engine check against the published FlyWire numbers:
    connections: Zenodo 10676866 (Dorkenwald et al. 2024), annotations: flyconnectome/flywire_annotations (Schlegel et al. 2024)
"""
import base64
import hashlib
import sys
import urllib.request

from .paths import DATA

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "malecns": [
        ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314, "UKdxh3DFciDxYLpPQxq4ng=="),
        ("body-neurotransmitters-male-cns-v1.0.feather", 43282834, "PYQrEv5cSe763lKNfdJKHw=="),
        ("connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", 508025642, "ZgHUrQr6mf0D6wh5Ze8kIw=="),
    ],
    "flywire": [
        ("Supplemental_file1_neuron_annotations.tsv", 31718505, None,
         "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/"
         "Supplemental_file1_neuron_annotations.tsv"),
        ("proofread_connections_783.feather", 852022274, "f48f972d262323a102aed49af1396b8a",
         "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"),
    ],
}


def md5_ok(path, ref):
    if ref is None:
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == ref if len(ref) == 32 else base64.b64encode(h.digest()).decode() == ref


def fetch(url, dest, size):
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fly-brainloss"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def download(which=("malecns",)):
    for key in which:
        d = DATA / key
        d.mkdir(parents=True, exist_ok=True)
        for item in FILES[key]:
            name, size, md5 = item[:3]
            url = item[3] if len(item) > 3 else MALECNS_BASE + name
            dest = d / name
            if dest.exists() and dest.stat().st_size == size:
                print(f"  {name}: present")
            else:
                fetch(url, dest, size)
            if dest.stat().st_size != size or not md5_ok(dest, md5):
                sys.exit(f"{name}: size or MD5 mismatch, delete it and run again")
            print(f"  {name}: ok")
