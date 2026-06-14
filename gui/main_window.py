import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QPlainTextEdit, QGroupBox, QCheckBox, QFileDialog,
    QMessageBox, QRadioButton, QButtonGroup, QDoubleSpinBox,
    QSpinBox, QComboBox,
)
from PyQt5.QtCore import Qt

from gui.worker_thread import WorkerThread
from gui.viewer_thread import ViewerThread
from core.paths import ROOT_DIR
from pipeline.pipeline import PhotogrammetryPipeline
from utils.eval_chamfer import show_mesh_with_metrics

CONFIG_PATH = str(ROOT_DIR / "config" / "config.yaml")
TOTAL_STEPS = 9


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photogrammetry Pipeline")
        self.setMinimumSize(820, 680)
        self.worker = None
        self.viewer_thread = None

        self._build_ui()
        self._load_config_to_ui()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # Header
        title = QLabel("Photogrammetry Pipeline")
        title.setObjectName("title_label")
        subtitle = QLabel("COLMAP + OpenMVS · Sparse → Dense → Mesh → Texture")
        subtitle.setObjectName("subtitle_label")
        root.addWidget(title)
        root.addWidget(subtitle)

        # ── Paths ────────────────────────────────────────────────────────────
        path_group = QGroupBox("Input / Output")
        pg = QVBoxLayout(path_group)
        pg.setSpacing(6)

        # Mode toggle row
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Input type:")
        mode_lbl.setFixedWidth(110)
        self.radio_images = QRadioButton("Images")
        self.radio_images.setChecked(True)
        self.radio_video = QRadioButton("Video")
        self._input_mode_group = QButtonGroup(self)
        self._input_mode_group.addButton(self.radio_images, 0)
        self._input_mode_group.addButton(self.radio_video, 1)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.radio_images)
        mode_row.addSpacing(16)
        mode_row.addWidget(self.radio_video)
        mode_row.addStretch()
        pg.addLayout(mode_row)

        # Image folder row (hidden in Video mode)
        self._image_row, self.image_dir_edit = self._make_path_row(
            "Image Folder:", "Select folder containing photos",
            lambda: self._browse_folder(self.image_dir_edit),
        )
        pg.addWidget(self._image_row)

        # Video file row (hidden in Images mode)
        self._video_row, self.video_file_edit = self._make_path_row(
            "Video File:", "Select video file (mp4, avi, mov, mkv, …)",
            self._browse_video_file,
        )
        self._video_row.setVisible(False)
        pg.addWidget(self._video_row)

        # Frame extraction sub-group (hidden in Images mode)
        self._frame_group = QGroupBox("Frame Extraction")
        self._frame_group.setObjectName("frame_extraction_group")
        frame_layout = QHBoxLayout(self._frame_group)
        frame_layout.setSpacing(12)
        frame_layout.addWidget(QLabel("Interval:"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSuffix(" sec")
        self.interval_spin.setFixedWidth(90)
        self.interval_spin.setToolTip("Time gap between extracted frames (seconds)")
        frame_layout.addWidget(self.interval_spin)
        frame_layout.addSpacing(20)
        frame_layout.addWidget(QLabel("Max frames:"))
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 9999)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setFixedWidth(80)
        self.max_frames_spin.setSpecialValueText("unlimited")
        self.max_frames_spin.setToolTip("Maximum number of frames to extract (0 = unlimited)")
        frame_layout.addWidget(self.max_frames_spin)
        frame_layout.addStretch()
        self._frame_group.setVisible(False)
        pg.addWidget(self._frame_group)

        # Workspace row (always visible)
        self._workspace_row, self.workspace_dir_edit = self._make_path_row(
            "Workspace:", "Where to store intermediate & output files",
            lambda: self._browse_folder(self.workspace_dir_edit),
        )
        pg.addWidget(self._workspace_row)

        self.radio_images.toggled.connect(self._on_input_mode_changed)
        self.radio_video.toggled.connect(self._on_input_mode_changed)
        root.addWidget(path_group)

        # ── Settings ─────────────────────────────────────────────────────────
        settings_group = QGroupBox("Settings")
        sg = QVBoxLayout(settings_group)
        sg.setSpacing(8)

        # Row 1: checkboxes
        chk_row = QHBoxLayout()
        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(True)
        self.refine_check = QCheckBox("Refine Mesh")
        self.refine_check.setChecked(True)
        self.texture_check = QCheckBox("Texture Mesh")
        self.texture_check.setChecked(True)
        chk_row.addWidget(self.gpu_check)
        chk_row.addSpacing(20)
        chk_row.addWidget(self.refine_check)
        chk_row.addSpacing(20)
        chk_row.addWidget(self.texture_check)
        chk_row.addStretch()
        sg.addLayout(chk_row)

        # Row 2: Pairing + Matcher dropdowns
        combo_row = QHBoxLayout()

        pairing_lbl = QLabel("Pairing:")
        pairing_lbl.setFixedWidth(55)
        self.pairing_combo = QComboBox()
        self.pairing_combo.addItem("Exhaustive", "exhaustive")
        self.pairing_combo.addItem("Sequential", "sequential")
        self.pairing_combo.addItem("Vocab Tree", "vocab_tree")
        self.pairing_combo.setFixedWidth(150)
        self.pairing_combo.setToolTip(
            "Exhaustive: all-pairs matching — best quality for small datasets (<300 images)\n"
            "Sequential: neighbours only — ideal for video / ordered captures\n"
            "Vocab Tree: retrieval-based — scales to large unordered datasets"
        )

        matcher_lbl = QLabel("Matcher:")
        matcher_lbl.setFixedWidth(58)
        self.matcher_combo = QComboBox()
        self.matcher_combo.addItem("SIFT", "sift")
        self.matcher_combo.addItem("SuperPoint + LightGlue", "ml")
        self.matcher_combo.setFixedWidth(190)
        self.matcher_combo.setToolTip(
            "SIFT: classic handcrafted features — fast and reliable\n"
            "SuperPoint + LightGlue: learned features — more accurate on difficult scenes"
        )

        combo_row.addWidget(pairing_lbl)
        combo_row.addWidget(self.pairing_combo)
        combo_row.addSpacing(20)
        combo_row.addWidget(matcher_lbl)
        combo_row.addWidget(self.matcher_combo)
        combo_row.addStretch()
        sg.addLayout(combo_row)

        # Sequential-specific: frame overlap (shown only for Sequential)
        self._seq_opts = QWidget()
        seq_layout = QHBoxLayout(self._seq_opts)
        seq_layout.setContentsMargins(75, 0, 0, 0)
        seq_layout.addWidget(QLabel("Frame overlap:"))
        self.seq_overlap_spin = QSpinBox()
        self.seq_overlap_spin.setRange(1, 50)
        self.seq_overlap_spin.setValue(10)
        self.seq_overlap_spin.setFixedWidth(60)
        self.seq_overlap_spin.setToolTip("Number of neighbouring frames to match on each side (SequentialMatching.overlap)")
        seq_layout.addWidget(self.seq_overlap_spin)
        seq_layout.addStretch()
        self._seq_opts.setVisible(False)
        sg.addWidget(self._seq_opts)

        # Vocab tree-specific: file path (shown only for Vocab Tree)
        self._vtree_opts = QWidget()
        vt_layout = QHBoxLayout(self._vtree_opts)
        vt_layout.setContentsMargins(0, 0, 0, 0)
        vt_lbl = QLabel("Vocab tree file:")
        vt_lbl.setFixedWidth(110)
        self.vocab_tree_edit = QLineEdit()
        self.vocab_tree_edit.setPlaceholderText("Path to vocab_tree.bin (download from colmap.github.io)")
        vt_browse = QPushButton("Browse")
        vt_browse.setFixedWidth(72)
        vt_browse.clicked.connect(self._browse_vocab_tree)
        vt_layout.addWidget(vt_lbl)
        vt_layout.addWidget(self.vocab_tree_edit)
        vt_layout.addWidget(vt_browse)
        self._vtree_opts.setVisible(False)
        sg.addWidget(self._vtree_opts)

        self.pairing_combo.currentIndexChanged.connect(self._on_pairing_changed)
        root.addWidget(settings_group)

        # ── Executables ──────────────────────────────────────────────────────
        exe_group = QGroupBox("Executables (edit config/config.yaml to change)")
        eg = QVBoxLayout(exe_group)
        self.colmap_exe_edit = self._exe_row(eg, "COLMAP:")
        self.interface_exe_edit = self._exe_row(eg, "InterfaceCOLMAP:")
        self.densify_exe_edit = self._exe_row(eg, "DensifyPointCloud:")
        root.addWidget(exe_group)

        # ── Progress ─────────────────────────────────────────────────────────
        prog_group = QGroupBox("Progress")
        pl = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, TOTAL_STEPS)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.progress_bar)
        pl.addWidget(self.status_label)
        root.addWidget(prog_group)

        # ── Log ──────────────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        ll = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        ll.addWidget(self.log_view)
        root.addWidget(log_group, 1)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.save_cfg_btn = QPushButton("💾  Save Config")
        self.save_cfg_btn.clicked.connect(self._save_config)

        self.clean_btn = QPushButton("🗑  Clean Workspace")
        self.clean_btn.setObjectName("clean_btn")
        self.clean_btn.clicked.connect(self._on_clean_workspace)
        self.clean_btn.setToolTip("Delete all intermediate files from the workspace folder")

        self.abort_btn = QPushButton("⛔  Abort")
        self.abort_btn.setObjectName("abort_btn")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._on_abort)

        self.open_btn = QPushButton("📂  Open Output Folder")
        self.open_btn.setObjectName("open_btn")
        self.open_btn.clicked.connect(self._on_open_output)

        self.view_3d_btn = QPushButton("🔷  View 3D Result")
        self.view_3d_btn.setObjectName("view_btn")
        self.view_3d_btn.clicked.connect(self._on_view_3d)

        self.start_btn = QPushButton("▶  Start Reconstruction")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self._on_start)

        btn_row.addWidget(self.save_cfg_btn)
        btn_row.addWidget(self.clean_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.abort_btn)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.view_3d_btn)
        btn_row.addWidget(self.start_btn)
        root.addLayout(btn_row)

    def _make_path_row(self, label_text, placeholder, browse_slot):
        """
        Build a single path row (label + line-edit + Browse button) inside a
        QWidget wrapper so it can be shown/hidden as a unit.  Every row in the
        Input/Output group is built this way so they all use addWidget and
        share identical left-edge alignment.

        Returns (wrapper_widget, line_edit).
        """
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(110)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        btn = QPushButton("Browse")
        btn.setFixedWidth(72)
        btn.clicked.connect(browse_slot)
        layout.addWidget(lbl)
        layout.addWidget(edit)
        layout.addWidget(btn)
        return wrapper, edit

    def _exe_row(self, layout, label_text):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(160)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setStyleSheet("color: #6c7086;")
        row.addWidget(lbl)
        row.addWidget(edit)
        layout.addLayout(row)
        return edit

    # ── Config ───────────────────────────────────────────────────────────────

    def _load_config_to_ui(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            exes = cfg.get("executables", {})
            self.colmap_exe_edit.setText(exes.get("colmap", ""))
            self.interface_exe_edit.setText(exes.get("interface_colmap", ""))
            self.densify_exe_edit.setText(exes.get("densify", ""))
            paths = cfg.get("paths", {})
            self.workspace_dir_edit.setText(str(ROOT_DIR / paths.get("workspace_dir", "data/workspace")))
            settings = cfg.get("settings", {})
            self.gpu_check.setChecked(settings.get("use_gpu", True))
            self.refine_check.setChecked(settings.get("run_refine", True))
            self.texture_check.setChecked(settings.get("run_texture", True))

            # Matcher dropdown
            use_ml = settings.get("use_ml_features", False)
            self.matcher_combo.setCurrentIndex(self.matcher_combo.findData("ml" if use_ml else "sift"))

            # Pairing dropdown + conditional sub-rows
            pairing = settings.get("matching_method", "exhaustive")
            idx = self.pairing_combo.findData(pairing)
            self.pairing_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.seq_overlap_spin.setValue(int(settings.get("seq_overlap", 10)))
            self.vocab_tree_edit.setText(settings.get("vocab_tree_path", "") or "")
            self._on_pairing_changed()
        except Exception as e:
            self._log(f"[WARN] Could not load config: {e}")

    def _save_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            cfg["settings"]["use_gpu"] = self.gpu_check.isChecked()
            cfg["settings"]["run_refine"] = self.refine_check.isChecked()
            cfg["settings"]["run_texture"] = self.texture_check.isChecked()
            cfg["settings"]["use_ml_features"] = self.matcher_combo.currentData() == "ml"
            cfg["settings"]["matching_method"] = self.pairing_combo.currentData()
            cfg["settings"]["seq_overlap"] = self.seq_overlap_spin.value()
            cfg["settings"]["vocab_tree_path"] = self.vocab_tree_edit.text().strip()
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            self._log("Config saved.")
        except Exception as e:
            self._log(f"[ERROR] Could not save config: {e}")

    # ── File Dialogs ─────────────────────────────────────────────────────────

    def _browse_folder(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            edit.setText(d)

    def _browse_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.webm);;All Files (*)"
        )
        if path:
            self.video_file_edit.setText(path)

    def _browse_vocab_tree(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Vocab Tree File", "",
            "Vocab Tree Files (*.bin *.fbow);;All Files (*)"
        )
        if path:
            self.vocab_tree_edit.setText(path)

    # ── Settings slots ───────────────────────────────────────────────────────

    def _on_input_mode_changed(self):
        video_mode = self.radio_video.isChecked()
        self._image_row.setVisible(not video_mode)
        self._video_row.setVisible(video_mode)
        self._frame_group.setVisible(video_mode)

    def _on_pairing_changed(self):
        method = self.pairing_combo.currentData()
        self._seq_opts.setVisible(method == "sequential")
        self._vtree_opts.setVisible(method == "vocab_tree")

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def _on_start(self):
        workspace_dir = self.workspace_dir_edit.text().strip()
        if not workspace_dir:
            QMessageBox.warning(self, "Missing Workspace", "Please select a workspace directory.")
            return

        video_mode = self.radio_video.isChecked()

        if video_mode:
            video_path = self.video_file_edit.text().strip()
            if not video_path or not Path(video_path).is_file():
                QMessageBox.warning(self, "Missing Video", "Please select a valid video file.")
                return
            image_dir = None
            frame_interval = self.interval_spin.value()
            max_frames = self.max_frames_spin.value()
        else:
            image_dir = self.image_dir_edit.text().strip()
            if not image_dir or not Path(image_dir).is_dir():
                QMessageBox.warning(self, "Missing Input", "Please select a valid image folder.")
                return
            video_path = None
            frame_interval = 1.0
            max_frames = 0

        self._save_config()
        self._reset_ui()
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.clean_btn.setEnabled(False)
        self._log("Starting pipeline...")

        self.worker = WorkerThread(
            image_dir=image_dir or "",
            workspace_dir=workspace_dir,
            config_path=CONFIG_PATH,
            video_path=video_path,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.aborted.connect(self._on_aborted)
        self.worker.start()

    def _on_abort(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Abort Reconstruction",
                "Stop the reconstruction and clean the workspace?\n\nAll intermediate files will be deleted.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._log("\n⚠ Aborting — stopping current process...")
                self.abort_btn.setEnabled(False)
                self.worker.abort()

    def _on_clean_workspace(self):
        workspace_dir = self.workspace_dir_edit.text().strip()
        if not workspace_dir:
            QMessageBox.warning(self, "No Workspace", "Set a workspace directory first.")
            return

        reply = QMessageBox.question(
            self, "Clean Workspace",
            f"Delete all files in:\n{workspace_dir}\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                PhotogrammetryPipeline.clean_workspace(workspace_dir, self._log)
                self._log("✅ Workspace cleaned.")
                self.progress_bar.setValue(0)
                self.status_label.setText("Workspace cleaned. Ready.")
            except Exception as e:
                self._log(f"[ERROR] Failed to clean workspace: {e}")

    def _reset_ui(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.log_view.clear()

    # ── Signals ──────────────────────────────────────────────────────────────

    def _on_progress(self, step, msg):
        self.progress_bar.setValue(step)
        self.status_label.setText(f"[{step}/{TOTAL_STEPS}] {msg}")
        self._log(f"── [{step}/{TOTAL_STEPS}] {msg}")

    def _on_finished(self, output_dir):
        self.progress_bar.setValue(TOTAL_STEPS)
        self.status_label.setText("✅ Reconstruction complete!")
        self._log(f"\n✅ Done! Output saved to: {output_dir}")
        self._restore_buttons()
        self.output_dir = output_dir

    def _on_error(self, msg):
        self.status_label.setText("❌ Error — see log for details")
        self._log(f"\n❌ ERROR: {msg}")
        self._restore_buttons()
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _on_aborted(self):
        self.status_label.setText("⛔ Aborted.")
        self._log("\n⛔ Reconstruction aborted.")
        self._restore_buttons()

        workspace_dir = self.workspace_dir_edit.text().strip()
        if workspace_dir:
            reply = QMessageBox.question(
                self, "Clean Workspace?",
                "Reconstruction was aborted.\n\nClean the workspace now to remove partial files?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    PhotogrammetryPipeline.clean_workspace(workspace_dir, self._log)
                    self._log("🗑  Workspace cleaned.")
                    self.progress_bar.setValue(0)
                    self.status_label.setText("Aborted & cleaned. Ready.")
                except Exception as e:
                    self._log(f"[ERROR] Cleanup failed: {e}")

    def _restore_buttons(self):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.clean_btn.setEnabled(True)

    def _on_view_3d(self):
        # Resolve output dir: prefer last run's result, then workspace/output
        output_dir = getattr(self, "output_dir", None)
        if not output_dir or not Path(output_dir).exists():
            workspace = self.workspace_dir_edit.text().strip()
            if workspace:
                candidate = Path(workspace) / "output"
                if candidate.exists():
                    output_dir = str(candidate)

        if not output_dir or not Path(output_dir).exists():
            QMessageBox.warning(
                self, "No Output Folder",
                "No output folder found.\n\n"
                "Set a workspace directory whose 'output' subfolder contains "
                "textured_mesh.obj (and optionally dense_point_cloud.ply).",
            )
            return

        mesh_path = Path(output_dir) / "textured_mesh.obj"
        if not mesh_path.exists():
            mesh_path = Path(output_dir) / "textured_mesh.ply"
        if not mesh_path.exists():
            QMessageBox.warning(
                self, "No Mesh Found",
                f"No mesh file found in:\n{output_dir}\n\n"
                "Expected: textured_mesh.obj or textured_mesh.ply",
            )
            return

        pcd_path = Path(output_dir) / "dense_point_cloud.ply"
        if pcd_path.exists():
            # Full evaluation with Chamfer / RMSE metrics
            self.view_3d_btn.setEnabled(False)
            self.status_label.setText("Computing metrics...")
            self.viewer_thread = ViewerThread(output_dir)
            self.viewer_thread.log.connect(self._log)
            self.viewer_thread.finished.connect(self._on_metrics_done)
            self.viewer_thread.error.connect(self._on_metrics_error)
            self.viewer_thread.start()
        else:
            # No point cloud — open viewer without metrics
            self._log("[3D Viewer] dense_point_cloud.ply not found — showing mesh without metrics.")
            show_mesh_with_metrics(str(mesh_path), {})

    def _on_metrics_done(self, metrics: dict, mesh_path: str):
        cd = metrics.get("chamfer_distance", 0.0)
        rmse_val = metrics.get("rmse", 0.0)
        self.status_label.setText(
            f"Chamfer: {cd:.6f} m  |  RMSE: {rmse_val:.6f} m  — opening viewer..."
        )
        self.view_3d_btn.setEnabled(True)
        show_mesh_with_metrics(mesh_path, metrics)

    def _on_metrics_error(self, msg: str):
        self.status_label.setText("❌ Metric computation failed — see log")
        self._log(f"\n❌ [3D Viewer] {msg}")
        self.view_3d_btn.setEnabled(True)
        QMessageBox.critical(self, "Evaluation Error", msg)

    def _log(self, msg):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _on_open_output(self):
        path = getattr(self, "output_dir", None) or self.workspace_dir_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No Folder", "Set a workspace directory first.")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "Folder Not Found", f"Folder does not exist:\n{path}")
            return
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
