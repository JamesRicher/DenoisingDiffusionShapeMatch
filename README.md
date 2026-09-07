# DenoisingDiffusionShapeMatch

Non-rigid shape correspondence by denoising diffusion over sparse assignment matrices,
with a geodesic message-passing denoiser.

## Setup

```bash
conda env create -f environment.yml
conda activate diffusion-shapematch
```

Datasets are located by `paths.py`, which checks `SHAPEMATCH_DATA_ROOT`, then
`<repo>/../../data`, then `<repo>/../data`:

```
data/
  FAUST_r/    SCAPE_r/    SHREC19_r/    SMAL_r/    DT4D_r/
```

## Training

```bash
python train.py -c configs/final/faust.yaml
```

Configs: `faust.yaml`, `scape.yaml`, `faust_scape.yaml`, `smal.yaml`, `dt4d.yaml`.
Output goes to `experiments/final/<name>/`.

## Inference

```bash
python evaluate.py -c configs/final/faust.yaml \
    --checkpoint experiments/final/faust/models/final.pth \
    --eval_tag faust --save_maps
```

Cross-dataset evaluation overrides the test set:

```bash
python evaluate.py -c configs/final/faust.yaml \
    --checkpoint experiments/final/faust/models/final.pth \
    --set datasets.test.name=Scape_r \
    --set datasets.test.type=SparsePairScapeDataset \
    --eval_tag scape
```

Add `--set bp=null` to disable the belief-propagation post-process. Results are written to
`experiments/final/<run>/results/<eval_tag>/`.

## Acknowledgement

Data is sourced and preprocessed with the ULRSSM code. This repository is largely based on the ULRSSM codebase in structure and conventions. The DiffusionNet feature extractor is ported from the same codebase, which is in turn based on the original implementation.

```bibtex
@article{cao2023unsupervised,
  title   = {Unsupervised Learning of Robust Spectral Shape Matching},
  author  = {Cao, Dongliang and Roetzer, Paul and Bernard, Florian},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2023}
}
```
