# Pinned versions

| Component | Version | Commit or hash |
| --- | --- | --- |
| Container image | `nvcr.io/nvidia/isaac-lab:2.3.0` | sha256 of the `.sif`: `fc4f70553252f4196990250bfc78cf561635206cecb0fb62587a95d5adc3182b` |
| Isaac Sim | 5.1.0 | in the image |
| Isaac Lab (image) | 2.3.0 | in the image |
| LeIsaac | v0.4.0 + 1 commit | `24d3bcd3f1e4585740fc79921782c41617237812` |
| Isaac Lab (LeIsaac's pin) | v2.3.0 | `3c6e67bb5c7ada942a6d1884ab69338f57596f77` |
| XLeRobot | main, 2026-10-06 | `b017b5e6354bd9f61f4247a920c72622ca0aade0` |
| SO-ARM100 (official SO-101 URDF + meshes) | main, `Simulation/SO101` | `a758567c3978dfeefe282ede0500085a48fe8f78` |
| LeRobot | 0.4.2 | from PyPI, see `requirements.lock.txt` |
| torch / numpy | 2.7.0+cu128 / 1.26.0 | in the image |

LeIsaac's Isaac Lab submodule is only recorded here. At run time the Isaac Lab
inside the image is used.
