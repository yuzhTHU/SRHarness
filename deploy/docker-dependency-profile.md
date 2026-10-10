# Docker dependency installation profile

Measured on 2026-10-10 while building the Linux/amd64 `sr-harness:1.0.0`
image from `python:3.12-slim-bookworm`. The build used Python 3.12, pip 25.0.1,
an empty pip cache, and `HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:6789`.

## Phase timings

| Phase | Time |
|---|---:|
| Build-system setup and SRHarness metadata | 6.3 s |
| Dependency resolution and wheel metadata | 27.7 s |
| Download 101 wheels (193.0 MB) | 28.9 s |
| Build the SRHarness wheel | 0.7 s |
| Install and unpack all wheels | 31.8 s |
| `pip check` | 1.4 s |
| Complete pip layer | 97.4 s |
| Export the Docker image layer | 7.1 s |

The resulting image is 981.2 MB. Installed Python distribution files account
for approximately 811.6 MiB.

## Largest wheel downloads

Times are the elapsed intervals between pip starting one wheel download and
starting the next. They therefore include a small amount of per-wheel pip
overhead and should be treated as approximate.

| Distribution | Wheel | Download time |
|---|---:|---:|
| pyarrow | 53.9 MB | 5.33 s |
| scipy | 35.3 MB | 3.80 s |
| numpy | 16.7 MB | 1.69 s |
| pandas | 10.8 MB | 1.06 s |
| matplotlib | 9.9 MB | 1.08 s |
| scikit-learn | 9.2 MB | 1.08 s |
| pillow | 6.9 MB | 0.81 s |
| sympy | 6.3 MB | 0.70 s |
| h5py | 5.4 MB | 0.56 s |
| fonttools | 5.4 MB | 0.63 s |
| cryptography | 4.8 MB | 0.53 s |
| hf-xet | 4.2 MB | 0.57 s |

## Largest installed distributions

| Distribution | Installed size |
|---|---:|
| pyarrow | 164.3 MiB |
| scipy | 132.6 MiB |
| sympy | 66.2 MiB |
| numpy | 65.2 MiB |
| pandas | 64.6 MiB |
| scikit-learn | 44.3 MiB |
| matplotlib | 33.8 MiB |
| fonttools | 26.7 MiB |
| pillow | 20.1 MiB |
| h5py | 17.4 MiB |
| openai | 14.3 MiB |
| cryptography | 14.9 MiB |
| hf-xet | 11.0 MiB |
| google-genai | 10.4 MiB |

## Packaging observations

- The current `tools` extra adds only about 7.7 MiB beyond the default
  dependency closure because PySR and PySINDy reuse NumPy, SciPy, pandas, and
  scikit-learn. This does not include Julia, which PySR may download on first
  use and which is substantially larger.
- `datasets` and dependencies unique to it account for about 191.6 MiB of the
  installed environment, led by pyarrow. In the current source tree,
  `datasets` and `h5py` are imported by `sr-harness benchmark`; they are the
  strongest candidates for a separate benchmark extra.
- Matplotlib and dependencies unique to it account for about 88.3 MiB. It is
  used by `sr_harness.utils.plot`, so making it optional would require the
  plotting API to fail clearly or load the dependency lazily.
- h5py contributes about 17.4 MiB beyond shared dependencies. Moving both
  h5py and datasets together would keep benchmark data loading self-contained.
- Moving only PySR, PySINDy, and pypdf between extras has little effect on this
  image's Python footprint. The PySR first-run Julia cost remains a separate
  operational concern.
