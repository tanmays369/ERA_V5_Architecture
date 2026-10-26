# Input text for the harness sample

`harness/build_sample.py` reads four files from this folder. They are not committed (see `../.gitignore`):
the English pack is diagnosis data that may contain personal data, and the Indic files are course material.

| File | What it is | Where to get it |
|---|---|---|
| `en-fineweb-edu.parquet` | 500 FineWeb-Edu documents (column `text`) | Your week-1 diagnosis pack |
| `hi_train.txt` | Cleaned Hindi Wikipedia text | Session 2 assignment, `assignment_2a/data/` |
| `bn_train.txt` | Cleaned Bengali text | Session 2 assignment, `assignment_2a/data/` |
| `te_train.txt` | Cleaned Telugu text | Session 2 assignment, `assignment_2a/data/` |

Any files of the same kind work; to use other names or places, change the paths in `../configs/data_mix.yaml`
(section `harness_sample`). `build_sample.py` writes the sha256 of every input file to
`work/sample/manifest.json`, so two people can check they built the same sample.
