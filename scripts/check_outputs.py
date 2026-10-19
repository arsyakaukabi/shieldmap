"""Compare the regenerated headline tables and Figs. 1-4 with the files published in this repository.

    python scripts/check_outputs.py

Prints one line per file: 'identical' (same bytes), 'identical (line endings)' (same bytes once CRLF, which pandas
writes on Windows, is read as LF; git stores these files with LF), or 'DIFFERENT'. Exit code 1 if any file differs.
"""
from __future__ import annotations

import sys

from _common import ROOT, sha256_bytes

EXPECTED = {
 "data/phase4b/headline_table_v3.csv": "235b5f33908839d4a09e2e51047c79d796809b5b5e58e12edbd380ea0a44f10d",
 "data/phase4d/headline_table_v4.csv": "240b96928dab57dcc0397dc6c4e57a0b5503c3d50f603921bf79552d96dcd007",
 "paper/figures/Fig1.pdf": "0e50efb3235d149212831f32d2a6befde4a17c95bed3005d737fbbbb33d892f9",
 "paper/figures/Fig1.png": "9c23e44e5860ac484a7a8d82fd82a8512a3e0d7f85704ca18d9443d56d0ec0ba",
 "paper/figures/Fig1.tif": "f277bedae440166e67b638973fb3d5ede397650be0c4194b1278ce6e75d5790c",
 "paper/figures/Fig2.pdf": "c22c2f9f8de03882a968e4cb86073829b530acd8c2b7d512cf520bfad2828b30",
 "paper/figures/Fig2.png": "9770da278f868ac22029bae66d4930c356a1d77f99fc74a8b782cc34ddf73c7b",
 "paper/figures/Fig2.tif": "9add94e612563f5eaca2646396e6bde02a96927b75ab6e17a3e3ed9796965809",
 "paper/figures/Fig3.pdf": "f03539c0ada645f9fd00cdabeebd4897cb98769a096882966f885e50745a0281",
 "paper/figures/Fig3.png": "12d07265caf2edbef4ef37d973b1c85c895f28b2e863d2eec85fc936b1f24f15",
 "paper/figures/Fig3.tif": "ad8d9302ab2682bf0730dac1fd07e3058e7f6560067fd1c870f250762efc6e89",
 "paper/figures/Fig4.pdf": "2c80f328f6748433902573c6be24ac04b3eb08bfe0b0dbb76ccc1b24a89fd7e8",
 "paper/figures/Fig4.png": "b5bbb86fa44e7d444eeb69a6221977a17a9c201ad040eae22ab5a5f7ab45c856",
 "paper/figures/Fig4.tif": "2b020365d50dd9b40b45845e978f211caa6ada399ece2cc005f0fce8698bf1e4"
}


def main() -> int:
    bad = 0
    for rel, h in EXPECTED.items():
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING     {rel}")
            bad += 1
            continue
        b = p.read_bytes()
        if sha256_bytes(b) == h:
            print(f"identical   {rel}")
        elif sha256_bytes(b.replace(b"\r\n", b"\n")) == h:
            print(f"identical (line endings)  {rel}")
        else:
            print(f"DIFFERENT   {rel}")
            bad += 1
    print("all outputs identical" if not bad else f"{bad} file(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
