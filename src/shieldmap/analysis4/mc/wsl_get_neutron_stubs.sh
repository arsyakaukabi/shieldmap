#!/usr/bin/env bash
# OpenMC 0.16 loads incident-neutron data for every nuclide in a material even in a
# photon-only fixed-source run (the neutron data are read but never sampled: the source emits photons only).
# Extract from the same verified archive only the neutron files of the nuclides of the MC materials.
set -uo pipefail
D="$HOME/endf"; F="$D/endfb80.tar.xz"
ELS="Pb Ir Hf O Bi F La Pr Sb Pd Os I Tl Au"
pat=$(for e in $ELS; do printf 'neutron/%s[0-9]+(_m[0-9])?\.h5$|' "$e"; done)
pat=${pat%|}
grep -E "$pat" "$D/endfb80.tar.xz.listing.txt" > "$D/neutron_members.txt"
echo "members: $(wc -l < $D/neutron_members.txt)"; cat "$D/neutron_members.txt"
tar -xJf "$F" -C "$D" -T "$D/neutron_members.txt"
echo "extracted_utc $(date -u +%FT%TZ)"
du -sh "$D/endfb-viii.0-hdf5/neutron"
echo NEUTRON_DONE
