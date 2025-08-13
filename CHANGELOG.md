
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-14

### Added
- Initial public release of datacollapse library
- Core data collapse fitting functions:
  - `fit_data_collapse()` for standard fitting without finite-size correction
  - `fit_data_collapse_fse()` for fitting with finite-size correction
  - `fit_data_collapse_fse_robust()` for robust finite-size correction with grid search
  - `collapse_transform()` for transforming data to collapse coordinates
- Spline-based universal function representation with smoothing penalty
- Multiple optimizer support (Nelder-Mead, Powell, combined)
- Multi-start random restarts for robust optimization
- Bootstrap uncertainty estimation
- Normalization support for finite-size corrections
- Progress callback interface for long-running fits
- Command-line interface via `cli.py`
- Comprehensive examples in `examples/` directory
- Unit tests with pytest
- GitHub Actions CI/CD pipeline
- Bilingual documentation (English/Chinese) with language toggle
- Professional README with data collapse visualization images
- Quadratic demo data generation for realistic physics examples
- Complete PyPI packaging configuration
- MIT license

### Documentation
- Comprehensive root README with installation, examples, and best practices
- Library API documentation with detailed parameter descriptions
- Visual demonstrations of data collapse before/after effects
- Performance notes and error handling guidelines
- Troubleshooting guide for common issues

### Features
- Support for heteroscedastic error weighting (1/σ²)
- Robust handling of finite-size correction parameter bounds (c < 0)
- Graceful fallback when scipy unavailable
- Reproducible results via random_state control
- Memory-efficient linear spline basis construction
- Second-difference smoothing penalty for universal function

### Technical
- Pure Python implementation with minimal dependencies
- Source layout packaging structure
- Optional dependencies for plotting and CLI features
- Python 3.9+ compatibility
- Comprehensive type hints and documentation
- Professional project metadata and classifiers
