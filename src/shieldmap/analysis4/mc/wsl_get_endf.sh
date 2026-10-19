#!/usr/bin/env bash
# One bulk download of the official OpenMC ENDF/B-VIII.0 HDF5 library
# (https://openmc.org/data/, "Official Data Libraries", ENDF/B-VIII.0, .tar.xz link).
# Resumable (curl -C -), SHA-256 + MD5 recorded, then ONLY photon/ data (photoatomic +
# atomic relaxation, plus the library's cross_sections.xml) are extracted; the neutron and
# thermal-scattering data are never extracted. The archive is deleted after hashing+extraction
# unless KEEP_ARCHIVE=1.
# Run detached:  setsid nohup bash wsl_get_endf.sh > ~/endf/get.log 2>&1 &
set -uo pipefail
URL="https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz"
EXPECTED_BYTES=3383607420
D="$HOME/endf"
mkdir -p "$D"
cd "$D"
F="$D/endfb80.tar.xz"
echo "start_utc $(date -u +%FT%TZ)"
for i in $(seq 1 60); do
  have=$( [ -f "$F" ] && stat -c %s "$F" || echo 0 )
  if [ "$have" -ge "$EXPECTED_BYTES" ]; then break; fi
  echo "attempt $i have=$have $(date -u +%FT%TZ)"
  curl -fL --retry 5 --retry-delay 10 -C - -o "$F" "$URL" || echo "curl exit $?"
done
have=$(stat -c %s "$F")
echo "download_finished_utc $(date -u +%FT%TZ) bytes=$have"
[ "$have" -eq "$EXPECTED_BYTES" ] || { echo "SIZE MISMATCH"; exit 2; }
sha256sum "$F" | tee "$D/endfb80.tar.xz.sha256"
md5sum "$F" | tee "$D/endfb80.tar.xz.md5"
tar -tJf "$F" > "$D/endfb80.tar.xz.listing.txt"
echo "listing entries: $(wc -l < $D/endfb80.tar.xz.listing.txt)"
PH=$(grep -E '/photon/|cross_sections.xml$' "$D/endfb80.tar.xz.listing.txt")
tar -xJf "$F" -C "$D" $PH
echo "extracted_utc $(date -u +%FT%TZ)"
find "$D" -name '*.h5' -path '*photon*' | wc -l
du -sh "$D"/*/ 2>/dev/null
if [ "${KEEP_ARCHIVE:-0}" != "1" ]; then rm -f "$F"; echo "archive deleted"; fi
echo ENDF_DONE
