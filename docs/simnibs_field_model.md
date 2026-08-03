# The SimNIBS field model — a FEM `gamma` for the JR nodes

**Status:** implemented, but the FEM solve fails. The code path is complete and tested; the registration gate passes, but SimNIBS crashes silently on Windows during KSP setup (an out-of-memory or PETSc failure). The FEM results tables below are therefore empty.

`gamma_model: simnibs` reads a precomputed SimNIBS FEM leadfield instead of the homogeneous
Coulomb kernel of [`tes_field_geometry.md`](tes_field_geometry.md) §9. It exists to answer the
question §9.4 deferred: **how much of the analytical model's spatial pattern is an artefact of
ignoring the skull and CSF?** It is a validity check, not a replacement — `analytical` remains the
default and no config, dataset or fitted predictor changes.

Source of truth:

- Loader: [`src/neuro/connectome.py`](../src/neuro/connectome.py)
  (`_simnibs_gamma_fn`, `centres_to_mni_ras`)
- Generators: [`scripts/export_simnibs_geometry.py`](../scripts/export_simnibs_geometry.py),
  [`scripts/generate_simnibs_leadfield.py`](../scripts/generate_simnibs_leadfield.py)
- Tests: [`tests/test_connectome.py`](../tests/test_connectome.py) (`test_simnibs_*`,
  `test_centres_to_mni_ras`)

## 1. Why two scripts

SimNIBS 4.6 ships **cp311 wheels only**; this project is Python 3.13. SimNIBS therefore cannot be
a dependency, and the work splits at the interpreter boundary:

```text
export_simnibs_geometry.py    uv,             py3.13, TVB      ->  geometry.npz  (MNI RAS)
generate_simnibs_leadfield.py simnibs_python, py3.11, SimNIBS  ->  gamma_*.npz
```

The split is drawn so the frame conversion — the single most bug-prone thing in this repo, see
§1b of the geometry note — lives in `neuro.connectome.centres_to_mni_ras`, under ruff, ty and
pytest, rather than in the script that no CI can import.

## 2. Setup and running it

```powershell
# 1. SimNIBS 4.6.0 -- run simnibs_installer_windows.exe from
#    https://github.com/simnibs/simnibs/releases/latest   (bundles its own py3.11)
simnibs_python -c "import simnibs; print(simnibs.__version__)"

# 2. Head model. Preferred: Ernie Extended (head *and shoulders*, MRI+CT), because standard
#    meshes stop at the neck and are inaccurate for exactly the extracephalic montage we use
#    (Van Hoornweder et al. 2024, Imaging Neuroscience).
#      https://osf.io/6qv2z/download   -> external/ernie_extended/     (CC BY-NC 4.0)
#    Fallback: the stock example dataset, m2m_ernie.
#      https://github.com/simnibs/example-dataset/releases/latest/download/simnibs4_examples.zip

# 3. Export the connectome geometry (project venv)
uv run python scripts/export_simnibs_geometry.py --out data/simnibs/geometry.npz

# 4. Registration only -- the gate. Costs seconds; run it before paying for any FEM.
simnibs_python scripts/generate_simnibs_leadfield.py --dry-run `
  --geometry data/simnibs/geometry.npz `
  --m2m external/ernie_extended/m2m_ernie_extended --out data/simnibs/gamma_ernie.npz

# 5. The FEM: one solve per stimulating electrode
simnibs_python scripts/generate_simnibs_leadfield.py `
  --geometry data/simnibs/geometry.npz `
  --m2m external/ernie_extended/m2m_ernie_extended `
  --electrodes TP9 CP5 --return-label EX_NECK `
  --out data/simnibs/gamma_ernie.npz
```

Then in a config, with the `simnibs_scale` the generator prints:

```yaml
connectome:
  target_electrode: [TP9, CP5, EX_NECK]
  gamma_model: simnibs
  leadfield_path: data/simnibs/gamma_ernie.npz
  simnibs_quantity: phi        # or e_normal
  simnibs_scale: 1.0           # from the generator's recommendation
```

No such config is checked in: `data/` and `*.npz` are gitignored, so a shipped config would break
[`tests/test_example_configs.py`](../tests/test_example_configs.py) on a clean checkout.

## 3. Registration, and the gate

`mni2subject_coords` warps the 76 region centres and the 16384 surface vertices from MNI RAS into
the head model's space. The failure mode that matters is not a crash but a **mirror**: a flipped
warp leaves every region present, every field plausible, and every number wrong.

So the generator re-runs §1b's independent witness in subject space and *raises* before touching
the FEM. A scalp channel's largest EEG lead-field entry must belong to a region physically near
that channel; the mean distance must stay under 50 mm (§1b measured 28.6 mm in the connectome
frame), and left-labelled regions must average negative RAS *x*.

| check | analytical reference | SimNIBS subject space |
| --- | --- | --- |
| mean electrode → own lead-field peak | 28.6 mm | 41.7 mm |
| left / right mean *x* | −/+ | −25.7 / +28.8 mm |
| `CP5 → lPCI`, `T7 → lA2`, `O1 → lV2` | 19 / 22 / 11 mm | 34.0 / 26.0 / 49.5 mm |

## 4. The two quantities

Both are emitted; `simnibs_quantity` selects one.

`phi` — the FEM potential averaged over brain nodes within 8 mm of each region centre, in mV per
mA. Same semantics as the analytical kernel, so it is the clean A/B: only the conductivity
structure changes. Like the analytical kernel it is a potential, so it has no absolute zero and
remains well-defined only under `sum(u) = 0` (§9.1).

`e_normal` — the cortical-normal E-field, area-averaged over each region's patch of SimNIBS's
middle grey-matter surface, in V/m per mA. Gauge-invariant, and the quantity that actually
polarises pyramidal membranes. §9.4 called this "the right thing".

The normal is never invented. TVB's 76-region parcellation is **entirely surface-based** —
verified in this repo: the region centres are exactly the centroids of their
`regionMapping_16k_76` patches (max deviation 5.6e-7 mm) and every region has support (min 29
vertices; `lHC` 54, `lAMYG` 135). So each region inherits the normal of its own cortical patch.
**Caveat:** for `lHC` and `lAMYG` that patch is parahippocampal/entorhinal sheet, not hippocampal
lamination. Since the EZ is mesial temporal, this is the weakest link in `e_normal` here — and the
reason `phi`, which needs no orientation, ships alongside it.

## 5. Calibration

The NPZ stays physical; the mA→mV constant lives in the config as `simnibs_scale`, so the artifact
survives recalibration and serves both quantities.

The generator derives its recommendation from the analytical model's dose anchor (§6 of the
geometry note): with −1 mA at the first stimulating electrode against the return, the mean drive
over `lHC, lPHC, lAMYG, lTCI, lTCV` is **−1.4681 mV** (reproduced exactly against the shipped
`analytical` model). Setting `simnibs_scale` to match it makes the dose–response tables directly
comparable, so what SimNIBS contributes is the *spatial pattern* rather than an overall gain.

The generator refuses to divide if cathodal current does not drive the EZ negative — that would
mean SimNIBS's `E_normal` sign convention (its middle-GM interpolation negates the normal
component) is inverted relative to expectation, and it must say so rather than silently flip.

| | analytical | `phi` | `e_normal` |
| --- | --- | --- | --- |
| EZ mean drive, −1 mA (raw units) | −1.4681 mV | *(to fill)* | *(to fill)* |
| derived `simnibs_scale` | 1.0 | *(to fill)* | *(to fill)* |

## 6. What to compare once it runs

1. **Spatial**: per-electrode Pearson and Spearman correlation of the `phi` rows against
   `analytical`; pairwise row cosine; zero-sum authority `s0(P·G^T)/s0(G^T)` on
   `[TP9, CP5, EX_NECK]`, against §2's 0.334.
2. **Depth dependence** — the headline claim of §9.4. Drive against distance-from-electrode,
   `1/r` versus the FEM. This is where skull and CSF shunting either show up or do not.
3. **Dose–response**: §6's table (0.1 / 0.2 / 0.35 / 0.5 / 1.0 mA) recomputed under each gamma.
4. **`phi` versus `e_normal`**: their correlation, and whether the EZ ranking survives.
5. **The falsifiable prediction.** §9.1 states that a *real* neck return drives a
   superior→inferior current, so inferior mesial-temporal structures see more field than the
   homogeneous `1/r` monopole allows — and therefore that §9.3's "no region is ever driven
   anodally" property should **not** survive a real extracephalic pad. If it does survive, either
   the mesh is truncated at the neck or the reduction is over-smoothing.
6. **One open-loop suppression run per gamma** at the calibrated scale. §9.8 concluded that
   forecast error does not predict suppression and that only the closed-loop outcome is
   trustworthy; without this the note is a correlation table, not a validation.

## 7. Caveats

- Ernie is not the TVB connectome's subject. The nonlinear MNI warp carries an unquantified error
  that the §1b witness bounds but does not remove. (41.7 mm vs the analytical model's 28.6 mm
  in its native frame — it passes the < 50 mm gate, but this error bounds how much any downstream
  comparison is worth).
- At 76 regions the parcellation is coarser than the field variation either model produces, so
  much of the added fidelity averages away (§9.4's third bullet still stands).
- Whether Ernie Extended or the truncated stock `ernie` was used changes the return electrode's
  meaning entirely; the generator records it in the NPZ's `provenance` field.
- Basis superposition is exact only because every run meshes the whole montage and varies only the
  channel currents. Changing the montage means regenerating. Note: SimNIBS 4.6.0 on Windows was
  found to crash on 0 A channels, requiring a fallback to a 2-electrode montage, which sacrifices
  exact superposition for solver stability.
- The SimNIBS solver (PETSc/hypre/pardiso) fails silently during KSP setup on the Windows test
  environment, likely due to memory limits or a DLL crash. The FEM tables (§5, §6) remain
  unpopulated as a result.
