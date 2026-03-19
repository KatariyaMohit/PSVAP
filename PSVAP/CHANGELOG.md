# PSVAP Changelog — Living Record

This file mirrors Section 12 of the PSVAP Software Development Master Plan.
Add a new entry **at the top** of the table every time you make a significant
change (including adding a dependency, creating a new file, or modifying a
public API).

| Date       | Author         | Phase   | Description of Change                                                                 |
|-----------|----------------|---------|---------------------------------------------------------------------------------------|
| 2026-03-13| Group F / AI   | Phase 0 | Comprehensive Phase 0 review & fix: fixed LammpsParser inheritance (now extends BaseParser), fixed controller.py SystemMetadata construction bug, rewrote main_window.py with QMenuBar + QSplitter + QTabWidget, cleaned export_panel, added gui/dialogs/ (file_open_dialog, settings_dialog, about_dialog), gui/widgets/ (atom_table_widget, plot_widget), visualization stubs (structure_renderer, interaction_renderer, plot_renderer, viewport_filters), io/exporter stub, plugins/sandbox + api + examples, docs/ directory, core/constants.py, test_selection.py, sample.pdb + sample.gro fixtures, added residue_id/chain_id/resname to selection.py, fixed pyproject.toml package discovery, made Residue/Chain mutable. |
| 2026-03-13| Group F / AI   | Phase 0 | Added Phase 0 skeleton: core SystemModel + selection parser, controller, viz engine, GUI shell, LAMMPS parser, and smoke tests. |
| 2026-03-13| Group F / AI   | Phase 0 | Initialized new PSVAP repository structure and top-level files per Master Plan v1.0. |
| 2026-03-13| Group F        | Phase 0 | Master Plan v1.0 created. Prototype code archived. New folder structure defined.     |

2026-03-13 | All (Group F) | Phase 1
- Created io/gromacs_parser.py  (MDAnalysis)
- Created io/pdb_parser.py      (Biopython)
- Created io/mmcif_parser.py    (Gemmi)
- Created io/amber_parser.py    (MDAnalysis)
- Created io/dcd_parser.py      (MDAnalysis)
- Created io/xyz_parser.py      (pure Python)
- Created io/mol_parser.py      (RDKit)
- Updated io/base_parser.py     (detect_parser supports all formats)
- Updated app/loader_worker.py  (uses detect_parser, not LammpsParser directly)
- Updated gui/main_window.py    (file dialogs show all supported formats)
- Added tests/test_parsers_phase1.py + 4 fixture files
Tag: v0.2-multiformat
2026-03-15Group FPhase 2core/atom.py: Added `resname: str2026-03-15Group FPhase 2analysis/sequence.py: Fixed extract_sequence to read atom.resname instead of atom.name. Previously always returned "X" for every residue.2026-03-15Group FPhase 2analysis/alignment.py: Added align_trajectory(model, reference_frame, atom_indices) function. Called by analysis_panel._run_align_trajectory(). Also changed rmsd_matrix to use relative import from .rmsd import rmsd.2026-03-15Group FPhase 2analysis/rmsd.py: Changed rmsd_after_superimpose to use relative import from .alignment import kabsch_rmsd instead of absolute from PSVAP.analysis.alignment.2026-03-15Group FPhase 2analysis/geometry.py: Fixed ramachandran to read atom.resname for residue name instead of atom.name.2026-03-15Group FPhase 2tests/test_rmsd.py: Written from scratch. 9 test classes, 35 tests covering rmsd, rmsd_trajectory, rmsf, rmsf_per_residue, rmsd_after_superimpose, kabsch_rmsd, superimpose, superimpose_trajectory, align_trajectory, rmsd_matrix.
| 2026-03-15 | Group F | Phase 2 | gui/main_window.py: Fixed render mode menu — added QActionGroup(exclusive=True) for radio-button behaviour, removed "BONDS ONLY" mode, fixed lambda to ignore triggered bool. _set_render_mode now shows confirmation in status bar. |

2026-03-15Group FPhase 3Created analysis/interactions.py — Features 12 & 21: H-bond, salt bridge, halogen bond, pi-stack, hydrophobic contact, clash detection.2026-03-15Group FPhase 3Created analysis/surface.py — Features 9 & 18: Shrake-Rupley SASA, per-residue SASA, surface patch classification.2026-03-15Group FPhase 3Created tests/test_interactions.py — 8 test classes, 28 tests for interactions and surface modules.2026-03-15Group FPhase 3Replaced visualization/interaction_renderer.py stub with full InteractionRenderer using named PyVista actors.2026-03-15Group FPhase 3Replaced visualization/plot_renderer.py stub with PlotRenderer text formatters.2026-03-15Group FPhase 3Updated gui/panels/analysis_panel.py — added INTERACT and SURFACE tabs with all compute slots.2026-03-15Group FPhase 3Updated gui/panels/modeling_panel.py — label updated to "PHASE 4+".

2026-03-15Group FPhase 4io/pdb_parser.py: Added resname=resname to Atom constructor in _build_atoms. PDB atoms now carry residue name for mutation engine, Ramachandran, and sequence extraction.2026-03-15Group FPhase 4Created modeling/mutation_engine.py — Feature 7: point mutations, residue listing, PDB writer.2026-03-15Group FPhase 4Created modeling/alanine_scan.py — Feature 11: systematic alanine scanning with interaction-based hot-spot scoring.2026-03-15Group FPhase 4Created modeling/structure_prep.py — Feature 17: structure QC report, remove waters/HETATM, cap termini, renumber residues.2026-03-15Group FPhase 4Created modeling/solvation.py — Feature 10: TIP3P/SPC-E water box construction, ion count estimation.2026-03-15Group FPhase 4Replaced gui/panels/modeling_panel.py stub with full PREP/MUTATE/ALA SCAN/SOLVATE panel.2026-03-15Group FPhase 4Created tests/test_modeling.py — 9 test classes, 38 tests covering all Phase 4 modules.