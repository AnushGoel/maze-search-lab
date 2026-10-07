# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-07

### Added
- `mazesearch` package: breadth-first, depth-first and A* search on grid
  mazes, with parsing of plain or gzip-compressed maze files.
- Five more algorithms: uniform-cost search, weighted A*, greedy best-first
  search, bidirectional BFS, and A* with landmark (ALT) lower bounds.
- Three seeded maze generators (recursive backtracker, Prim's algorithm,
  binary tree) and transformations: adding loops, removing walls, moving S/E.
- Analysis tools: structure statistics, the A* "must expand" profile, and an
  exit census that prices every possible exit from two traversals.
- Reproducible experiment pipeline (`scripts/run_experiments.py`, eight
  experiments, resumable by stage) and its datasets in `data/results`.
- Seven-page Streamlit app with live solving, animated search races and the
  full study.
- Test suite, GitHub Actions workflow, citation file and research notes.
