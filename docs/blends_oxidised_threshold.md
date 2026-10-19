# Elemental blends, the fully oxidised filter and the transparency threshold

*Each number keeps its source row of the headline table (`data/phase4b/headline_table_v3.csv`) in square brackets. Unless stated otherwise all values are narrow-beam, polymorph rule R2, coherent scattering included, frontier kind best_clean, and prices are B4-internal with thickness at the measured-density point.*

The elemental blend reference line is the best binary or ternary mass mixture of the garment elements Sn, Sb, Bi, W, Ba and Gd, computed with the mixture rule and the narrow-beam solver of the map.
At the reference beam the best blend (Sn, Ba and Gd in roughly equal parts) needs 0.45 % less areal mass than the compound frontier and 26 % less than Bi; the best binary blend (Sb and Gd) is 3.9 % above the frontier. [H180, H181, H188]
At RQR5 behind 0.25 mm Pb the best blend is 0.75 % below the frontier and 18 % below Bi; at RQR8 behind 0.35 mm Pb it is 2.0 % and 31 % below. [H182, H183, H184, H185]
Over the 36 variants the best blend, whose elements change with the variant, lies between −2.5 and +1.4 % of the frontier and 18–36 % below Bi. [H186:lo, H186:hi, H187:lo, H187:hi]
The line is narrow-beam; at the reference beam the Monte Carlo finds the Sn/Ba/Gd blend 2.3 % below the frontier crystal with transported fluorescence ([Monte Carlo spot check](monte_carlo.md)), and real garments add a polymer matrix. [H276]

Two filters form the fully oxidised tier (O/F anions only).
AMB-ANION admits fully oxidised compounds: the only anions are O and F, and the charge balance must be reached with positive oxidation states of every other element; it excludes intermetallics, cluster iodides, reduced oxides and NH₃·H₂O, and it is a necessary, not sufficient, condition for air stability.
AMB-ICSD3 requires at least three ICSD entries and hardly filters: it adds 0.97 % at the reference beam and 0 % at Cs-137 and Co-60 on both bases. [H207, H210, H211, H212, H213]
AMB-ANION adds 12 % at the reference beam and 10 % at RQR5 (leader CsAgO, 2 ICSD entries) and 14 % at RQR8 (CsSmO₂, 1 entry); both leaders are hygroscopic, so these values are lower bounds on the price of ambient stability. [H200, H201, H202]
AMB-ANION and AMB-ICSD3 together add 17 % at the reference beam, 23 % at RQR5 and 18 % at RQR8 (leaders Cs₈In₂O₇, Ba₃InRu₂O₉ and LaHoO₃). [H320, H321, H322]
At the radionuclide lines it adds 85 % (Cs-137) and 82 % (Co-60) to the thickness, with Pt₃O₄ as leader, and 3.3 % and 0.43 % to the areal mass. [H203, H205, H204, H206]
The best fully oxidised lanthanide compound with at least three ICSD entries is Ba₂SmSbO₆ (3 entries) at the reference beam, 21 % above the frontier and 11 % below Bi; at RQR5 it is Ba₄Tb(RuO₄)₃ (3 entries), 30 % above the frontier and 7.4 % above Bi; at RQR8 it is LaHoO₃ (4 entries), 18 % above the frontier and 17 % below Bi. [H214, H215, H216, H217, H218, H219]
Without the ICSD condition the best fully oxidised lanthanide compound at the reference beam is BaNdInO₄ (1 entry), which was not transported.

The transparency threshold moves the radionuclide leaders as follows.
At Cs-137 BiOF leads from 2.5 to 3.1 eV (135 %), with a GGA(+U) gap of 3.13 eV; from 3.2 eV HfO₂ leads (148 %) up to 4.0 eV. [stmt:H221, H221, H241, stmt:H235, H235, stmt:H231, H231]
At Co-60 HfO₂ leads at 130 % for every threshold from 2.5 to 4.0 eV. [H223, stmt:H223, stmt:H233]
On the areal-mass basis the Cs-137 price rises from 4.7 % at 2.5 eV to 8.1 % at 4.0 eV. [H222, stmt:H222, H232, stmt:H232]
Requiring a cubic crystal as well makes Lu₂O₃ the leader, at 167 % (Cs-137) and 148 % (Co-60) in thickness. [H237, H239]

## Blend search and the oxidation-state rule (for reproducibility)

**Blend search (`src/shieldmap/analysis4b/revision.py`, `_best_combo`).**
- Every binary and ternary subset of Sn, Sb, Bi, W, Ba and Gd is searched on a coarse 5 % mass-fraction simplex.
- The search is then refined locally around the coarse optimum: in 0.1 % steps for binaries, and in 0.25 % steps for
  the three best ternary combinations.
- The required areal mass is solved with the narrow-beam solver of the map for the same target transmission.

**AMB-ANION oxidation-state rule (`revision._states`, `revision._fully_oxidised`).**
- The only anions are O (−2) and F (−1).
- Every other element takes one of its admissible states:
  - its positive common oxidation states in pymatgen;
  - for d-block metals and lanthanides, also its positive ICSD-reported states (e.g. Pr⁴⁺, Tb⁴⁺, Cu⁺);
  - p-block low states seen only in ICSD are not admitted.
- Charge balance Σ nᵢ oxᵢ = 2 n_O + n_F must be reachable, with mixed valence allowed.
- Supplementary Table S1 states the same rule.
