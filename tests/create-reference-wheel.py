"""
Modify a hirola wheel so that it can be installed alongside regular hirola for
old vs new benchmarking
"""

import argparse
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("wheel")
options = parser.parse_args()

wheel = Path(options.wheel)

with zipfile.ZipFile(wheel) as zip:
    with zipfile.ZipFile(wheel.with_name(wheel.name.replace("hirola", "hirola_old")), "w") as out:
        for name in zip.namelist():
            out.writestr(name.replace("hirola", "hirola_old"), zip.read(name).replace(b"hirola", b"hirola_old"))
        print("Written", out.filename)
