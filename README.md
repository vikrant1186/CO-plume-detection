# Does a methane plume detector transfer to carbon monoxide?

Applying the automated plume-detection approach of **Schuit et al. (2023)** —
a CNN trained on TROPOMI CH4 scenes — to **TROPOMI CO** columns over
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

> **Does a plume-morphology detector trained on CH4 transfer to CO?**
The results are discussed in the "CO_plume_detection_plain_guide.pdf"

> **To see the figures (Plume score along with missing pixels) for each site:**
go through **figures** folder"


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
