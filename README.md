# DenoisingDiffusionShapeMatch

Non-rigid shape correspondence by denoising diffusion over sparse assignment matrices,
with a geodesic message-passing denoiser.

## Setup

```bash
conda env create -f environment.yml
conda activate jrr25-shapematch
```

Datasets are located by `paths.py`, which checks `SHAPEMATCH_DATA_ROOT`, then
`<repo>/../../data`, then `<repo>/../data`:

```
data/
  FAUST_r/    SCAPE_r/    SHREC19_r/    SMAL_r/    DT4D_r/
```

## Training

```bash
python train.py -c configs/final/faust_mpnn_512_final_cold_co.yaml
```

Configs: `faust`, `scape`, `faust_scape`, `smal`, `dt4d` (all `*_mpnn_512_final_cold_co.yaml`).
Output goes to `experiments/final/<name>/`.

## Inference

```bash
python evaluate.py -c configs/final/faust_mpnn_512_final_cold_co.yaml \
    --checkpoint experiments/final/faust_mpnn_512_final_cold_co/models/final.pth \
    --eval_tag faust --save_maps
```

Cross-dataset evaluation overrides the test set:

```bash
python evaluate.py -c configs/final/faust_mpnn_512_final_cold_co.yaml \
    --checkpoint experiments/final/faust_mpnn_512_final_cold_co/models/final.pth \
    --set datasets.test.name=Scape_r \
    --set datasets.test.type=SparsePairScapeDataset \
    --eval_tag scape
```

Add `--set bp=null` to disable the belief-propagation post-process. Results are written to
`experiments/final/<run>/results/<eval_tag>/`.

## Acknowledgement

Data is sourced and preprocessed with the ULRSSM code, and this repository follows its
conventions: the remeshed dataset variants and their splits, the cached spectral operators
and their hashing scheme, and the area-normalised geodesic error used for evaluation. The
DiffusionNet feature extractor is ported from the same codebase.

```bibtex
@article{cao2023unsupervised,
  title   = {Unsupervised Learning of Robust Spectral Shape Matching},
  author  = {Cao, Dongliang and Roetzer, Paul and Bernard, Florian},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2023}
}
```
