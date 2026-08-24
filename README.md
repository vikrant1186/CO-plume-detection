# Does a methane plume detector transfer to carbon monoxide?

Applying the automated plume-detection approach of **Schuit et al. (2023)** —
a CNN trained on TROPOMI CH₄ scenes — to **TROPOMI CO** columns over
industrial and urban sources, and testing whether spatial plume morphology is
species-agnostic.

**Vikrant Tomar** · ARIES, Nainital · [vikrant1186.github.io](https://vikrant1186.github.io)

---

## The question

Schuit et al. detect methane super-emitters with a two-stage pipeline: a small
CNN scores 32×32-pixel TROPOMI scenes for plume-like *morphology*, then a
support-vector classifier separates real plumes from retrieval artefacts using
41 engineered covariates. The CNN is trained on the **xch4 channel alone** —
it never sees anything species-specific beyond the mixing-ratio field itself.

If the scenes are normalised to dimensionless anomaly maps before training,
the network cannot know which gas it is looking at. So:

> **Does a plume-morphology detector trained on CH₄ transfer to CO?**

Both outcomes are useful:

| Outcome | What it means |
|---|---|
| It transfers | Morphology is species-agnostic — a CO detection system can warm-start from the existing methane model instead of labelling a CO training set from scratch. |
| It does not | You have quantified what needs retraining and why: CO plumes sit on a larger, more variable background, are broader and weaker relative to it, and are pervasively confused with biomass burning. |

The second result is not a failure. It is a specification.

## What happened

**It transfers over strong sources, but only once you control for how much of
the scene the satellite actually saw.**

One year (2019), 18 evenly spaced days, the same days at every site, 26 sites,
335 usable tiles. The CH₄-trained model was applied to CO **without any
retraining**. On the methane benchmark itself the reconstruction reaches
recall 0.969 / precision 0.963 (published: 0.956 / 0.942), so the starting
point is sound.

| Group | n sites | detected, all tiles | detected, tiles ≥90% complete |
|---|---|---|---|
| Indian steel | 13 | 52% | **66%** |
| Indian cities | 3 | 26% | 39% |
| European steel | 7 | 12% | 33% |
| Background controls | 3 | 15% | 21% |

Indian steel vs. background controls, matched on scene completeness:
**AUC 0.81, p = 1×10⁻⁴**. Stacking every day per site and measuring the excess
at the facility gives Indian steel **+0.69**, Indian cities +0.47, European
steel +0.24, background **−0.17** — the controls go negative, which is what a
working control should do.

The within-site seasonal test is the strongest single control: the same Indian
plants score far higher in the dry season than in the monsoon (AUC 0.75,
p = 0.013 restricted to the clearest scenes), while the background sites show
no seasonal difference.

**Two things had to be fixed to get there, and both are worth reading before
you reuse this code.** See [Known issues](#known-issues), and
[`interpretation/CO_plume_findings_explained.docx`](interpretation/) for the
whole story in plain language.

![scene completeness confound](figures/analysis/rank_diagnosis.png)

## Known issues

Both were found by auditing results that looked good, and both are the kind of
thing that generalises to any satellite plume-detection pipeline.

### 1. Zero-filled gaps couple the detector to cloud cover

`normalise_scene` fills missing pixels with 0 after normalising. A patch of
zeros is perfectly flat, carries no plume morphology, and drags the score down.
The consequence is not subtle: across the 26 site means, **mean plume score vs.
fraction of valid pixels gives Spearman ρ = +0.87** (ρ = +0.81 within the
Indian sites alone, where tile geometry is near-constant; ρ = +0.58 per scene
across all 335 tiles).

So the raw per-facility ranking is largely a ranking of clear-sky data volume.
Coastal sites are the clearest casualty — the CO SWIR retrieval barely works
over water, so JSW Dolvi (~10 Mt/yr) and Mumbai land at the *bottom* of the
list with ~65% valid pixels. Two background controls outscore Duisburg.

**Do not quote the per-facility ranking.** Quote the completeness-matched
comparison above. The proper fix is to mask gaps out of the convolution rather
than fill them, or to fill with a local median so a hole is not flat; that
change is not yet made.

### 2. Tiles were not centred on the facility away from the equator (fixed)

`_tile_region` widens the longitude half-width by 1/cos(lat) so the box is
square in kilometres. `_sample` then reprojected with `scale=PIXEL_KM*1000`,
and **Earth Engine converts a metre `scale` inside a geographic CRS using
metres-per-degree at the equator** — 0.0629° per pixel at every latitude. The
widening was therefore never undone by the sampling grid: the returned array is
32 rows by `32/cos(lat)` columns. 34 columns at Bhilai (21°N), **51 at Duisburg
(51.5°N)**.

`_pad_to_tile` then kept `arr[:32, :32]` — the *western* 32 columns — leaving
the plant at column ~25.5 instead of ~15.5. Every European site was scored,
stacked and centre-excess'd **about 65 km west of its own steelworks**.

Fixed in `fetch_co_tiles.py` by pinning an explicit `crsTransform` instead of a
metre `scale`, plus a centre-crop guard in `_pad_to_tile`. Re-measuring the
existing tiles at the true facility position lifts European steel from +0.14 to
+0.24 and drops the background controls from −0.02 to −0.17.

> ⚠️ **The committed `co_detections.json` and figures predate this fix.** The
> code here is corrected; the stored outputs are not. Re-running
> `fetch_co_tiles.py` is the first thing to do with this repository.

![tile geometry bug](figures/analysis/tile_geometry_bug.png)

## Why CO is harder than CH₄

Three things change when you swap the species, and the results here should be
read against them:

1. **Background.** CH₄ has a smooth, well-characterised background. CO's varies
   strongly with season, transport and fire activity, so the per-scene
   normalisation is doing more work — and can flatten a real plume.
2. **Fires.** Biomass burning is a pervasive, mobile, plume-shaped CO source.
   Any global industrial CO catalogue built from 2018 onwards is contaminated
   by it unless the two are separated. This is the main obstacle, and it is
   handled explicitly below.
3. **Sensitivity.** The TROPOMI CO retrieval has different vertical sensitivity
   and cloud handling from the CH₄ retrieval, so artefact morphology differs —
   which means the stage-two artefact classifier is *not* transferable even if
   the stage-one CNN is.

## Separating industrial plumes from fires

Van der Velde et al. (2021) show that the **TROPOMI CO : NO₂ column ratio** is
an observed proxy for combustion efficiency — low for smouldering biomass
burning, high for efficient flaming combustion. This repository uses that ratio
together with co-located **MODIS MaxFRP** as an independent fire check, so every
detection carries a `fire_suspected` flag rather than being silently counted as
industrial.

## Pipeline

```
src/get_training_data.py   download Schuit et al. labelled scenes (Zenodo, CC-BY 4.0)
src/dataset.py             normalisation, 8× dihedral augmentation, torch Dataset
src/model.py               the CNN (reconstruction — see caveat below)
src/train.py               train on CH₄, benchmark against the published scores
src/fetch_co_tiles.py      pull 32×32 TROPOMI CO + NO₂ + MODIS FRP tiles via GEE
src/predict_co.py          run the CH₄-trained model on CO, flag fire contamination
src/browse.py              one contact sheet per site, filtered by score
src/selftest.py            end-to-end check on synthetic plumes, no downloads
```

## Interpreting the output

`interpretation/` holds four documents. The first is the one to read if you
only read one:

| File | What it is |
|---|---|
| `CO_plume_findings_explained.docx` | **Plain-language walkthrough of the results**, including both known issues. No prior knowledge assumed. |
| `CO_plume_detection_plain_guide.pdf` | How the method works, step by step, with flowcharts. |
| `report_1_the_code.pdf` | What every source file does and why. |
| `report_3_the_results.pdf` | The numbers, figure by figure. |

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the pipeline works before downloading anything (~1 min)
python src/selftest.py

# 2. get the labelled methane scenes (~418 MB)
python src/get_training_data.py --out data

# 3. train and compare against the published benchmark
python src/train.py --data-dir data --epochs 30

# 4. pull CO tiles over the target facilities
earthengine authenticate
python src/fetch_co_tiles.py --targets targets/facilities.csv \
    --start 2019-01-01 --end 2019-12-31 --out data/co_tiles.npz

# 5. the actual experiment
python src/predict_co.py --model plume_cnn.pt --tiles data/co_tiles.npz
```

## Targets

`targets/facilities.csv` has three groups:

- **European steel plants** from Leguijt et al. (2025), which published
  TROPOMI-derived CO emissions alongside E-PRTR reported values — the positive
  control. If the detector cannot see these, it is not working.
- **Indian steel plants and cities.** India is the world's second-largest crude
  steel producer with no E-PRTR-equivalent facility reporting, so satellite
  estimates there cannot be validated the way Europe's were. This is where a
  global catalogue is hardest to trust.
- **Negative controls** — remote ocean, desert, and a clean high-altitude site.
  If these score high, the detector is finding artefacts.

13 of the 20 steel sites have been checked against Global Energy Monitor's
plant database. One (JSPL Angul) was 11.4 km out — 1.6 pixels — and has been
corrected. The remaining 7 are marked `[UNVERIFIED]` in the CSV and should be
checked before any quantitative use.

## Honest caveats

These are stated up front because the people most likely to read this built the
instrument and wrote the original method.

- **The architecture is a reconstruction.** Schuit et al. describe "two
  convolutional blocks followed by two fully connected layers" on 32×32
  single-channel scenes. Layer widths, kernel sizes, dropout and optimiser are
  not fully specified, so `src/model.py` is a reasonable reconstruction, not the
  published network. This reproduces their *approach*, not their model.
- **The normalisation scheme is a choice, not theirs.** A robust per-scene
  standardisation (median and IQR) is what makes cross-species transfer possible
  at all. The paper's exact scheme should be checked and matched.
- **This runs on the harmonised L3 product, not L2.** `COPERNICUS/S5P/OFFL/L3_CO`
  in Earth Engine is regridded, pre-filtered by qa_value, and carries no
  averaging kernel or retrieval diagnostics. That is a deliberate speed
  trade-off for a prototype. The same pipeline runs on native L2 with a HARP
  regridding step in front, which is how it would have to be done properly.
- **Detection is not quantification.** Nothing here estimates an emission rate.
  Going from a detected plume to a source rate needs cross-sectional flux or an
  inversion with transport modelling — the step Leguijt et al. do with a WRF
  ensemble.
- **The detector is sensitive to missing data, and that is not a footnote.**
  See [Known issues](#known-issues). Any comparison between two groups of sites
  that differ in cloud cover — which is to say, almost any comparison between
  regions — has to be matched on scene completeness first.
- **The synthetic self-test passes easily**, and that is expected. Synthetic
  negatives have no coherent spatial structure, so the discrimination is trivial.
  Real negatives contain retrieval artefacts that genuinely look like plumes —
  which is precisely why the published method needs a second stage.

## References

- Schuit, B. J., Maasakkers, J. D., Bijl, P., et al. (2023). Automated detection
  and monitoring of methane super-emitters using satellite data. *Atmos. Chem.
  Phys.* **23**, 9071–9098. https://doi.org/10.5194/acp-23-9071-2023
  Training data: https://doi.org/10.5281/zenodo.13903869 (CC-BY 4.0)
- Leguijt, G., Maasakkers, J. D., Denier van der Gon, H. A. C., Segers, A. J.,
  Borsdorff, T., van der Velde, I. R., and Aben, I. (2025). Comparing space-based
  to reported carbon monoxide emission estimates for Europe's iron and steel
  plants. *Atmos. Chem. Phys.* **25**, 555–574.
  https://doi.org/10.5194/acp-25-555-2025
- van der Velde, I. R., van der Werf, G. R., Houweling, S., Eskes, H. J.,
  Veefkind, J. P., Borsdorff, T., and Aben, I. (2021). Biomass burning combustion
  efficiency observed from space using measurements of CO and NO₂ by TROPOMI.
  *Atmos. Chem. Phys.* **21**, 597–616. https://doi.org/10.5194/acp-21-597-2021
- Sentinel-5P TROPOMI CO, via Google Earth Engine
  (`COPERNICUS/S5P/OFFL/L3_CO`). Operational CO retrieval developed by SRON.
