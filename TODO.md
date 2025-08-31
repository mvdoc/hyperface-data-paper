# Hyperface Data Paper Repository - Restructuring Tasks

## Completed Tasks ✅

### Repository Structure & Package Setup
- [x] Create new directory structure (src/, docs/)
- [x] Rename `budapestcode` package to `hyperface`
- [x] Update package `__init__.py` with new version info
- [x] Create `pyproject.toml` with UV dependency management
- [x] Update all imports from `budapestcode` to `hyperface` in scripts

### Path Updates
- [x] Update all scripts to use `data/derivatives/` structure
- [x] Change `outputs/fmriprep` → `data/derivatives/fmriprep`
- [x] Change `outputs/datapaper/` → `data/derivatives/qa/`
- [x] Update quality assurance scripts paths
- [x] Sample notebook path updates (one notebook as example)

### Jupyter Book Setup
- [x] Create `docs/_config.yml` with book configuration
- [x] Create `docs/_toc.yml` with notebook organization
- [x] Create `docs/index.md` with project overview

## Remaining Tasks 📋

### Data Integration
- [ ] Add data submodule placeholder (when ready): `git submodule add <datalad-repo-url> data`
- [ ] Test scripts with actual hyperface data structure
- [ ] Verify all QA metrics output to correct `data/derivatives/qa/` locations

### Notebook Completion
- [ ] Update remaining 7 notebooks with new data paths:
  - [ ] `2020-03-19_plot-motion-parameters.ipynb`
  - [ ] `2020-03-29_compute-tsnr.ipynb`
  - [ ] `2020-04-19_compute-isc.ipynb`
  - [ ] `2020-04-20_plot-isc.ipynb`
  - [ ] `2020-06-08_make-event-files.ipynb`
  - [ ] `2020-06-11_plot-tsnr-fsaverage.ipynb`
  - [ ] `2020-07-07_compute-outliers-and-median-motion.ipynb`
- [ ] Rename notebooks to match hyperface timeline/naming conventions
- [ ] Test notebook execution with hyperface data

### Script Completion
- [ ] Update remaining scripts in other directories:
  - [ ] `scripts/hyperalignment_and_decoding/`
  - [ ] `scripts/preprocessing-fmri/`
  - [ ] `scripts/preprocessing-stimulus/`
  - [ ] `scripts/presentation/`
- [ ] Verify all output paths are correct for `data/derivatives/qa/`
- [ ] Test script execution

### Documentation & Polish
- [ ] Update main README.md for hyperface project
- [ ] Add installation instructions using UV
- [ ] Document data submodule usage
- [ ] Update .gitignore for new structure
- [ ] Create requirements for building Jupyter Book
- [ ] Add example commands for running QA pipeline

### Testing & Validation
- [ ] Test UV environment setup: `uv venv && source .venv/bin/activate && uv pip install -e .`
- [ ] Test notebook execution in new structure
- [ ] Test Jupyter Book build: `jupyter-book build docs/`
- [ ] Verify all output directories are created correctly
- [ ] Test QA pipeline end-to-end with hyperface data

### Future Enhancements
- [ ] Add continuous integration for book building
- [ ] Add automated testing for QA scripts
- [ ] Consider adding pre-commit hooks for code quality
- [ ] Add badges for documentation build status

## Key Changes Made

1. **Package Structure**: Moved from `code/budapestcode/` to `src/hyperface/`
2. **Data Organization**: Follows BIDS derivatives convention with `data/derivatives/`
3. **Dependency Management**: Switched from conda requirements.txt to UV + pyproject.toml
4. **Documentation**: Set up Jupyter Book for web publishing
5. **Path Updates**: All analysis scripts now point to correct BIDS derivative locations

## Notes

- The `data/` directory will be added as a git submodule/datalad dataset later
- All notebooks need systematic updating - one example completed
- Scripts are partially updated - main QA scripts done, others need review
- Jupyter Book is configured but will need testing once notebooks are ready
- UV dependency management ready but needs testing with actual environment