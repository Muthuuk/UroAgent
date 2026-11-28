## [Unreleased]
### Added
- Click-to-load samples in the imaging tab: three computer-drawn CT teaching phantoms (SPC balloon, bladder wall mass, dilated ureters with a stone) and three fictional reports (cystoscopy, PI-RADS 4 prostate MRI, CT urogram).
### Fixed
- The report reader now tells a stone in the kidney or ureter apart from a bladder stone.

# Changelog

## 0.2.0 - 2026-11-27
- New welcome home page ("Welcome to the UroAgentic AI App") with animated 3D buttons that open each feature.
- Colab notebook loads the MedGemma image model automatically on a GPU runtime, reading the Hugging Face key from a Colab secret (HF_TOKEN).
- Imaging tab explains how to switch on the image model when it is not loaded.
- The audit log's model version now follows the package version.

## 0.1.0 - 2026-11-25
First archived release (Zenodo DOI), as described in the paper: patient symptom assistant, clinician agent with memory,
imaging and report analysis (including bladder and prostate cancer checks), score calculator, MCP clinical tools and evaluation.
