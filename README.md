# Model Compass — GitHub release staging folder

This folder is prepared as the code-and-manuscript portion of the Model Compass release. To publish it, create or select a GitHub repository and push the **contents of this folder**. The paper's GitHub URL can then be added to its Code availability statement.

## Contents

- `server/lib/`: frozen scoring and telemetry implementation used by the paper's analysis harnesses.
- `scratch/*.js` and `scratch/*.py`: analysis and verification scripts.
- `scratch/intents.frozen.json`: recorded parser outputs used to avoid re-calling the hosted parser.
- `docs/`: the current main LaTeX source, Springer Nature class file, and build script.

## Data is intentionally not bundled

This staging copy omits the raw Agent Arena snapshots, the frozen model registry (including benchmark-derived fields), and generated result tables. The upstream artifact notes that these third-party inputs are not cleared for public redistribution. See `DATA_NOTICE.md` for the required inputs and their expected paths. Do not add them to a public repository unless redistribution rights have been confirmed or the sources have been replaced with redistributable data.

The analysis scripts require the authorized inputs listed in `DATA_NOTICE.md`; without those files, the empirical analyses cannot be reproduced from this folder alone. No MIT license is asserted for this bundle because the code and data licensing review is still pending. The Springer Nature class has its own license notice in `docs/sn-jnl.cls`.

## Build the paper

From this directory, run `bash docs/build-paper.sh` in an environment with a LaTeX engine. The class file is included beside the manuscript.

## Before making the repository public

1. Confirm the redistribution terms for every external dataset and derived output.
2. Select a license that applies only to material the authors are entitled to license; add required third-party notices.
3. Add the repository URL to the manuscript after the GitHub upload.
