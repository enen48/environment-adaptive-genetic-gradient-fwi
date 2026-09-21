# OpenFWI sample data license and attribution

The two `openfwi_flatvel_a_*.npy` files are selected velocity models from
**OpenFWI FlatVel-A**, by Chengyuan Deng, Shihang Feng, Hanchen Wang, Xitong Zhang,
Peng Jin, Yinan Feng, Qili Zeng, Yinpeng Chen, and Youzuo Lin.

**Data license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0).** The project's code license does not replace
this data license.

- [Official license declaration](https://openfwi-lanl.github.io/docs/data.html)
- [License terms](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [Legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
- [Original project](https://openfwi-lanl.github.io/)
- [Paper](https://arxiv.org/abs/2111.02926)

Redistribution must retain attribution and the license link, be noncommercial,
identify changes, and apply the required share-alike terms to adaptations. Do not
apply extra restrictions that prevent the permissions granted by this license.
These data carry no endorsement from their original authors.

Changes made here: retain samples 0–63 of `FlatVel_A/model/model1.npy` and samples
0–7 of `FlatVel_A/model/model49.npy`, respectively; no resampling, normalization,
augmentation, or numerical modification. The original physical velocities remain
in m/s. This is a small smoke-test selection, not an official new benchmark.

The download transport was the **unofficial**
[ashynf/OpenFWI mirror](https://huggingface.co/datasets/ashynf/OpenFWI).
The authoritative data license is the original OpenFWI declaration above.
Hashes, pinned mirror URLs, shapes, and exact indices are recorded in
[provenance.json](provenance.json). Hashes were checked against the mirror's file
pages; independent byte comparison with the official Google Drive copy was not
performed.

Suggested citation: Deng et al. (2022), *OpenFWI: Large-Scale Multi-Structural
Benchmark Datasets for Full Waveform Inversion*, NeurIPS Datasets and Benchmarks.

