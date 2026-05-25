#!/usr/bin/env python3
"""
MainWindow definition extracted from the original single-file GUI.
All GUI wiring, widgets, and the MainWindow class live here.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QFormLayout, QSlider, QSpinBox, QDoubleSpinBox,
    QProgressBar, QMessageBox, QLineEdit, QComboBox, QCheckBox, QToolButton, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from .worker import Worker
from .analysis_panel import AnalysisPanel
from .collapsible_box import CollapsibleBox
from utils import qpixmap_from_path
from .theme import apply_dark_palette
import numpy as np
import configparser
import math
import os
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Load config.ini ---
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "..", "config.ini"))

        def get(section, key, default, cast=str):
            try:
                val = config.get(section, key)
                return cast(val)
            except Exception:
                return default

        def getbool(section, key, default):
            try:
                return config.getboolean(section, key)
            except Exception:
                return default

        # --- Window ---
        self.setWindowTitle("图像检测绕过工具 V1.4 Alpha 1")
        self.setMinimumSize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)

        # Left: previews & file selection
        left_v = QVBoxLayout()
        main_h.addLayout(left_v, 2)

        # Input/Output collapsible
        io_box = CollapsibleBox("输入/输出")
        left_v.addWidget(io_box)
        in_layout = QFormLayout()
        io_container = QWidget()
        io_container.setLayout(in_layout)
        io_box.content_layout.addWidget(io_container)

        self.input_line = QLineEdit()
        self.input_btn = QPushButton("选择输入图片")
        self.input_btn.setMinimumWidth(200)
        self.input_btn.setMinimumHeight(32)
        self.input_btn.clicked.connect(self.choose_input)

        self.ref_line = QLineEdit()
        self.ref_btn = QPushButton("选择白平衡参考图（可选）")
        self.ref_btn.setMinimumWidth(240)
        self.ref_btn.setMinimumHeight(32)
        self.ref_btn.clicked.connect(self.choose_ref)

        self.fft_ref_line = QLineEdit()
        self.fft_ref_btn = QPushButton("选择参考图（FFT、GLCM）（可选）")
        self.fft_ref_btn.setMinimumWidth(240)
        self.fft_ref_btn.setMinimumHeight(32)
        self.fft_ref_btn.clicked.connect(self.choose_fft_ref)

        self.output_line = QLineEdit()
        self.output_btn = QPushButton("选择输出路径")
        self.output_btn.setMinimumWidth(200)
        self.output_btn.setMinimumHeight(32)
        self.output_btn.clicked.connect(self.choose_output)

        in_layout.addRow(self.input_btn, self.input_line)
        in_layout.addRow(self.ref_btn, self.ref_line)
        in_layout.addRow(self.fft_ref_btn, self.fft_ref_line)
        in_layout.addRow(self.output_btn, self.output_line)

        # Previews
        self.preview_in = QLabel(alignment=Qt.AlignCenter)
        self.preview_in.setFixedSize(480, 300)
        self.preview_in.setStyleSheet("background:#121213;border:1px solid #2b2b2b;color:#bdbdbd;border-radius:6px")
        self.preview_in.setText("输入预览")

        self.preview_out = QLabel(alignment=Qt.AlignCenter)
        self.preview_out.setFixedSize(480, 300)
        self.preview_out.setStyleSheet("background:#121213;border:1px solid #2b2b2b;color:#bdbdbd;border-radius:6px")
        self.preview_out.setText("输出预览")

        left_v.addWidget(self.preview_in)
        left_v.addWidget(self.preview_out)

        # Actions
        actions_h = QHBoxLayout()
        self.run_btn = QPushButton("运行 — 处理图片")
        self.run_btn.clicked.connect(self.on_run)
        self.open_out_btn = QPushButton("打开输出文件夹")
        self.open_out_btn.clicked.connect(self.open_output_folder)
        actions_h.addWidget(self.run_btn)
        actions_h.addWidget(self.open_out_btn)
        left_v.addLayout(actions_h)
        
        # Task count setting
        task_layout = QFormLayout()
        self.task_count_spin = QSpinBox()
        self.task_count_spin.setRange(1, os.cpu_count() or 8)
        self.task_count_spin.setValue(get("General", "task_count", max(1, (os.cpu_count() or 4)//2), int))
        self.task_count_spin.setToolTip("并行处理任务数，建议设置为CPU核心数的一半")
        task_layout.addRow("并行任务数", self.task_count_spin)

        # Video memory buffer setting
        self.video_memory_gb_spin = QDoubleSpinBox()
        self.video_memory_gb_spin.setRange(0.1, 32.0)
        self.video_memory_gb_spin.setSingleStep(0.5)
        self.video_memory_gb_spin.setValue(get("General", "video_memory_gb", 2.0, float))
        self.video_memory_gb_spin.setToolTip("视频处理时使用的内存缓冲区大小（GB），留空则自动计算")
        self.video_memory_gb_spin.setSpecialValueText("自动")
        self.video_memory_gb_chk = QCheckBox("使用自定义视频内存")
        self.video_memory_gb_chk.setChecked(getbool("General", "video_memory_enabled", False))
        self.video_memory_gb_chk.stateChanged.connect(self._on_video_memory_toggled)
        
        video_memory_hbox = QHBoxLayout()
        video_memory_hbox.addWidget(self.video_memory_gb_chk)
        video_memory_hbox.addWidget(self.video_memory_gb_spin)
        task_layout.addRow("视频内存 (GB)", video_memory_hbox)

        # Video buffer frames setting (alternative)
        self.video_buffer_frames_spin = QSpinBox()
        self.video_buffer_frames_spin.setRange(16, 1024)
        self.video_buffer_frames_spin.setValue(get("General", "video_buffer_frames", 128, int))
        self.video_buffer_frames_spin.setToolTip("视频处理时内存中保留的最大帧数，留空则自动计算")
        self.video_buffer_frames_chk = QCheckBox("使用自定义帧数")
        self.video_buffer_frames_chk.setChecked(getbool("General", "video_buffer_frames_enabled", False))
        self.video_buffer_frames_chk.stateChanged.connect(self._on_video_buffer_frames_toggled)
        
        video_buffer_frames_hbox = QHBoxLayout()
        video_buffer_frames_hbox.addWidget(self.video_buffer_frames_chk)
        video_buffer_frames_hbox.addWidget(self.video_buffer_frames_spin)
        task_layout.addRow("视频缓冲帧数", video_buffer_frames_hbox)

        left_v.addLayout(task_layout)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_v.addWidget(self.progress)

        # Right: controls + analysis panels (with scroll area)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        main_h.addWidget(scroll_area, 3)

        scroll_widget = QWidget()
        right_v = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)

        # Auto Mode toggle
        self.auto_mode_chk = QCheckBox("启用自动模式")
        self.auto_mode_chk.setChecked(getbool("General", "auto_mode", False))
        self.auto_mode_chk.stateChanged.connect(self._on_auto_mode_toggled)
        right_v.addWidget(self.auto_mode_chk)

        # Auto Mode section collapsible
        self.auto_box = CollapsibleBox("自动模式")
        right_v.addWidget(self.auto_box)
        auto_layout = QFormLayout()
        auto_container = QWidget()
        auto_container.setLayout(auto_layout)
        self.auto_box.content_layout.addWidget(auto_container)

        strength_layout = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(get("AutoMode", "strength", 25, int))
        self.strength_slider.valueChanged.connect(self._update_strength_label)
        self.strength_label = QLabel(str(self.strength_slider.value()))
        self.strength_label.setFixedWidth(30)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)
        auto_layout.addRow("畸变强度", strength_layout)

        # Blend system
        self.blend_box = CollapsibleBox("颜色融合")
        right_v.addWidget(self.blend_box)
        blend_layout = QFormLayout()
        blend_container = QWidget()
        blend_container.setLayout(blend_layout)
        self.blend_box.content_layout.addWidget(blend_container)

        self.blend_chk = QCheckBox("启用颜色融合")
        self.blend_chk.setToolTip("颜色融合将相似颜色的区域合并为同一种颜色")
        self.blend_chk.setChecked(getbool("Blend", "enabled", False))
        blend_layout.addRow(self.blend_chk)

        self.blend_tolerance = QSpinBox()
        self.blend_tolerance.setRange(1, 100)
        self.blend_tolerance.setValue(get("Blend", "tolerance", 10, int))
        self.blend_tolerance.setToolTip("融合颜色容差（越小颜色越多）")
        blend_layout.addRow("颜色容差", self.blend_tolerance)

        self.blend_min_region = QSpinBox()
        self.blend_min_region.setRange(1, 1000)
        self.blend_min_region.setValue(get("Blend", "min_region", 50, int))
        self.blend_min_region.setToolTip("保留的最小区域大小（像素）")
        blend_layout.addRow("最小区域大小", self.blend_min_region)

        self.blend_max_samples = QSpinBox()
        self.blend_max_samples.setRange(1000, 1000000)
        self.blend_max_samples.setValue(get("Blend", "max_samples", 100000, int))
        self.blend_max_samples.setToolTip("K均值聚类采样的最大像素数（提升速度）")
        blend_layout.addRow("最大采样数", self.blend_max_samples)

        self.blend_n_jobs = QSpinBox()
        self.blend_n_jobs.setRange(1, os.cpu_count() or 4)
        self.blend_n_jobs.setValue(get("Blend", "n_jobs", os.cpu_count() or 4, int))
        self.blend_n_jobs.setToolTip("融合工作线程数（默认：CPU核心数）")
        blend_layout.addRow("工作线程数", self.blend_n_jobs)

        # AI Normalizer
        self.ai_norm_box = CollapsibleBox("AI归一化")
        right_v.addWidget(self.ai_norm_box)
        ai_layout = QFormLayout()
        ai_container = QWidget()
        ai_container.setLayout(ai_layout)
        self.ai_norm_box.content_layout.addWidget(ai_container)

        self.ns_chk = QCheckBox("启用AI归一化（需要PyTorch）")
        self.ns_chk.setToolTip("启用AI归一化，需要PyTorch")
        self.ns_chk.setChecked(getbool("AINormalizer", "enabled", False))
        ai_layout.addRow(self.ns_chk)

        self.ns_iterations_spin = QSpinBox()
        self.ns_iterations_spin.setRange(1, 10000)
        self.ns_iterations_spin.setValue(get("AINormalizer", "iterations", 500, int))
        self.ns_iterations_spin.setToolTip("Number of iterations for the AI Normalizer optimization.")
        ai_layout.addRow("迭代次数", self.ns_iterations_spin)

        self.ns_lr_spin = QDoubleSpinBox()
        self.ns_lr_spin.setDecimals(6)
        self.ns_lr_spin.setRange(0.000001, 0.1)
        self.ns_lr_spin.setSingleStep(0.0001)
        self.ns_lr_spin.setValue(get("AINormalizer", "learning_rate", 0.0003, float))
        self.ns_lr_spin.setToolTip("AI归一化优化的学习率")
        ai_layout.addRow("学习率", self.ns_lr_spin)

        self.ns_t_lpips_spin = QDoubleSpinBox()
        self.ns_t_lpips_spin.setDecimals(6)
        self.ns_t_lpips_spin.setRange(0.000001, 1.0)
        self.ns_t_lpips_spin.setSingleStep(0.0001)
        self.ns_t_lpips_spin.setValue(get("AINormalizer", "t_lpips", 0.04, float))
        self.ns_t_lpips_spin.setToolTip("时间加权LPIPS损失参数")
        ai_layout.addRow("时间LPIPS", self.ns_t_lpips_spin)

        self.ns_t_l2_spin = QDoubleSpinBox()
        self.ns_t_l2_spin.setDecimals(6)
        self.ns_t_l2_spin.setRange(0.000001, 1.0)
        self.ns_t_l2_spin.setSingleStep(0.00001)
        self.ns_t_l2_spin.setValue(get("AINormalizer", "t_l2", 3e-05, float))
        self.ns_t_l2_spin.setToolTip("时间加权L2损失参数")
        ai_layout.addRow("时间L2", self.ns_t_l2_spin)

        self.ns_c_lpips_spin = QDoubleSpinBox()
        self.ns_c_lpips_spin.setDecimals(6)
        self.ns_c_lpips_spin.setRange(0.000001, 1.0)
        self.ns_c_lpips_spin.setSingleStep(0.0001)
        self.ns_c_lpips_spin.setValue(get("AINormalizer", "c_lpips", 0.01, float))
        self.ns_c_lpips_spin.setToolTip("内容损失LPIPS权重")
        ai_layout.addRow("内容LPIPS", self.ns_c_lpips_spin)

        self.ns_c_l2_spin = QDoubleSpinBox()
        self.ns_c_l2_spin.setDecimals(6)
        self.ns_c_l2_spin.setRange(0.000001, 10.0)
        self.ns_c_l2_spin.setSingleStep(0.01)
        self.ns_c_l2_spin.setValue(get("AINormalizer", "c_l2", 0.6, float))
        self.ns_c_l2_spin.setToolTip("内容损失L2权重")
        ai_layout.addRow("内容L2", self.ns_c_l2_spin)

        self.ns_grad_clip_spin = QDoubleSpinBox()
        self.ns_grad_clip_spin.setDecimals(6)
        self.ns_grad_clip_spin.setRange(0.000001, 1.0)
        self.ns_grad_clip_spin.setSingleStep(0.0001)
        self.ns_grad_clip_spin.setValue(get("AINormalizer", "grad_clip", 0.05, float))
        self.ns_grad_clip_spin.setToolTip("梯度裁剪阈值，用于稳定训练")
        ai_layout.addRow("梯度裁剪", self.ns_grad_clip_spin)

        # Parameters (Manual Mode) collapsible
        self.params_box = CollapsibleBox("参数（手动模式）")
        right_v.addWidget(self.params_box)
        params_layout = QFormLayout()
        params_container = QWidget()
        params_container.setLayout(params_layout)
        self.params_box.content_layout.addWidget(params_container)

        # New optional flags for processing steps
        self.noise_enable_chk = QCheckBox("启用高斯噪声")
        self.noise_enable_chk.setChecked(getbool("ManualParameters", "noise_enable", True))
        params_layout.addRow(self.noise_enable_chk)

        self.fft_enable_chk = QCheckBox("启用FFT频谱匹配")
        self.fft_enable_chk.setChecked(getbool("ManualParameters", "fft_enable", True))
        params_layout.addRow(self.fft_enable_chk)

        # FFT variant selector
        self.fft_variant_combo = QComboBox()
        self.fft_variant_combo.addItems(["v1 (Original)", "v2", "v3"])
        self.fft_variant_combo.setCurrentText(get("ManualParameters", "fft_variant", "v2"))
        self.fft_variant_combo.setToolTip("选择使用的傅里叶处理版本")
        params_layout.addRow("FFT版本", self.fft_variant_combo)

        self.perturb_enable_chk = QCheckBox("启用随机扰动")
        self.perturb_enable_chk.setChecked(getbool("ManualParameters", "perturb_enable", True))
        params_layout.addRow(self.perturb_enable_chk)

        # Noise-std
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.001)
        self.noise_spin.setValue(get("ManualParameters", "noise_std", 0.02, float))
        self.noise_spin.setToolTip("高斯噪声标准差（相对于255的比例）")
        params_layout.addRow("噪声标准差（0-0.1）", self.noise_spin)

        # Cutoff
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.01, 1.0)
        self.cutoff_spin.setSingleStep(0.01)
        self.cutoff_spin.setValue(get("ManualParameters", "cutoff", 0.25, float))
        params_layout.addRow("傅里叶截止（0-1）", self.cutoff_spin)

        # Fstrength
        self.fstrength_spin = QDoubleSpinBox()
        self.fstrength_spin.setRange(0.0, 1.0)
        self.fstrength_spin.setSingleStep(0.01)
        self.fstrength_spin.setValue(get("ManualParameters", "fstrength", 0.9, float))
        params_layout.addRow("傅里叶强度（0-1）", self.fstrength_spin)

        # Randomness
        self.randomness_spin = QDoubleSpinBox()
        self.randomness_spin.setRange(0.0, 1.0)
        self.randomness_spin.setSingleStep(0.01)
        self.randomness_spin.setValue(get("ManualParameters", "randomness", 0.05, float))
        params_layout.addRow("傅里叶随机性", self.randomness_spin)

        # Phase_perturb
        self.phase_perturb_spin = QDoubleSpinBox()
        self.phase_perturb_spin.setRange(0.0, 1.0)
        self.phase_perturb_spin.setSingleStep(0.001)
        self.phase_perturb_spin.setValue(get("ManualParameters", "phase_perturb", 0.08, float))
        self.phase_perturb_spin.setToolTip("相位扰动标准差（弧度）")
        params_layout.addRow("相位扰动（弧度）", self.phase_perturb_spin)

        # Radial_smooth
        self.radial_smooth_spin = QSpinBox()
        self.radial_smooth_spin.setRange(0, 50)
        self.radial_smooth_spin.setValue(get("ManualParameters", "radial_smooth", 5, int))
        params_layout.addRow("径向平滑（分箱）", self.radial_smooth_spin)

        # FFT_mode
        self.fft_mode_combo = QComboBox()
        self.fft_mode_combo.addItems(["auto", "ref", "model"])
        self.fft_mode_combo.setCurrentText(get("ManualParameters", "fft_mode", "auto"))
        params_layout.addRow("FFT模式", self.fft_mode_combo)

        # FFT_alpha
        self.fft_alpha_spin = QDoubleSpinBox()
        self.fft_alpha_spin.setRange(0.1, 4.0)
        self.fft_alpha_spin.setSingleStep(0.1)
        self.fft_alpha_spin.setValue(get("ManualParameters", "fft_alpha", 1.0, float))
        self.fft_alpha_spin.setToolTip("使用模型模式时1/f模型的Alpha指数")
        params_layout.addRow("FFT Alpha（模型模式）", self.fft_alpha_spin)

        # Perturb
        self.perturb_spin = QDoubleSpinBox()
        self.perturb_spin.setRange(0.0, 0.05)
        self.perturb_spin.setSingleStep(0.001)
        self.perturb_spin.setValue(get("ManualParameters", "perturb", 0.008, float))
        params_layout.addRow("像素扰动", self.perturb_spin)

        # Seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(get("ManualParameters", "seed", 0, int))
        params_layout.addRow("随机种子（0=无）", self.seed_spin)

        # AWB checkbox
        self.awb_chk = QCheckBox("启用自动白平衡（AWB）")
        self.awb_chk.setChecked(getbool("AWB", "enabled", False))
        self.awb_chk.setToolTip("勾选后应用AWB。如果选择了参考图，则使用参考图；否则使用灰世界AWB")
        params_layout.addRow(self.awb_chk)

        # Camera simulator toggle
        self.sim_camera_chk = QCheckBox("启用相机管道模拟")
        self.sim_camera_chk.setChecked(getbool("CameraSimulator", "enabled", False))
        self.sim_camera_chk.stateChanged.connect(self._on_sim_camera_toggled)
        params_layout.addRow(self.sim_camera_chk)

        # LUT support UI
        self.lut_chk = QCheckBox("启用LUT")
        self.lut_chk.setChecked(getbool("LUT", "enabled", False))
        self.lut_chk.setToolTip("启用对输出图片应用1D/.npy/.cube LUT")
        self.lut_chk.stateChanged.connect(self._on_lut_toggled)
        params_layout.addRow(self.lut_chk)

        self.lut_line = QLineEdit(get("LUT", "file", ""))
        self.lut_btn = QPushButton("选择LUT")
        self.lut_btn.setMinimumWidth(120)
        self.lut_btn.setMinimumHeight(32)
        self.lut_btn.clicked.connect(self.choose_lut)
        lut_box = QWidget()
        lut_box_layout = QHBoxLayout()
        lut_box_layout.setContentsMargins(0, 0, 0, 0)
        lut_box.setLayout(lut_box_layout)
        lut_box_layout.addWidget(self.lut_line)
        lut_box_layout.addWidget(self.lut_btn)
        self.lut_file_label = QLabel("LUT文件（png/.npy/.cube）")
        params_layout.addRow(self.lut_file_label, lut_box)

        self.lut_strength_spin = QDoubleSpinBox()
        self.lut_strength_spin.setRange(0.0, 1.0)
        self.lut_strength_spin.setSingleStep(0.01)
        self.lut_strength_spin.setValue(get("LUT", "strength", 1.0, float))
        self.lut_strength_spin.setToolTip("LUT融合强度（0.0=无效果，1.0=完整LUT）")
        self.lut_strength_label = QLabel("LUT强度")
        params_layout.addRow(self.lut_strength_label, self.lut_strength_spin)

        # Initially hide LUT controls and their labels
        self.lut_file_label.setVisible(False)
        lut_box.setVisible(False)
        self.lut_strength_label.setVisible(False)
        self.lut_strength_spin.setVisible(False)

        self._lut_controls = (self.lut_file_label, lut_box, self.lut_strength_label, self.lut_strength_spin)

        # Initialize video controls visibility
        self.video_memory_gb_spin.setEnabled(self.video_memory_gb_chk.isChecked())
        self.video_buffer_frames_spin.setEnabled(self.video_buffer_frames_chk.isChecked())

        # Texture Normalization collapsible group
        self.texture_box = CollapsibleBox("纹理归一化")
        right_v.addWidget(self.texture_box)
        texture_layout = QFormLayout()
        texture_container = QWidget()
        texture_container.setLayout(texture_layout)
        self.texture_box.content_layout.addWidget(texture_container)

        # GLCM checkbox
        self.glcm_chk = QCheckBox("启用GLCM归一化")
        self.glcm_chk.setChecked(getbool("TextureNormalization", "glcm_enabled", False))
        self.glcm_chk.setToolTip("使用FFT参考图启用GLCM归一化")
        texture_layout.addRow(self.glcm_chk)

        # GLCM distances
        self.glcm_distances_line = QLineEdit(get("TextureNormalization", "glcm_distances", "1"))
        self.glcm_distances_line.setToolTip("GLCM计算的距离列表，空格分隔（如 '1 2 3'）")
        texture_layout.addRow("GLCM距离", self.glcm_distances_line)

        # GLCM angles
        self.glcm_angles_line = QLineEdit(get("TextureNormalization", "glcm_angles", "0 0.785 1.571 2.356"))
        self.glcm_angles_line.setToolTip("GLCM的角度列表（弧度），空格分隔（如 '0 0.785 1.571 2.356'）")
        texture_layout.addRow("GLCM角度（弧度）", self.glcm_angles_line)

        # GLCM levels
        self.glcm_levels_spin = QSpinBox()
        self.glcm_levels_spin.setRange(2, 256)
        self.glcm_levels_spin.setValue(get("TextureNormalization", "glcm_levels", 256, int))
        self.glcm_levels_spin.setToolTip("GLCM的灰度级数")
        texture_layout.addRow("GLCM级数", self.glcm_levels_spin)

        # GLCM strength
        self.glcm_strength_spin = QDoubleSpinBox()
        self.glcm_strength_spin.setRange(0.0, 1.0)
        self.glcm_strength_spin.setSingleStep(0.01)
        self.glcm_strength_spin.setValue(get("TextureNormalization", "glcm_strength", 0.9, float))
        self.glcm_strength_spin.setToolTip("GLCM特征匹配强度（0.0=无效果，1.0=完整效果）")
        texture_layout.addRow("GLCM强度", self.glcm_strength_spin)

        # Camera simulator collapsible group
        self.camera_box = CollapsibleBox("相机模拟器选项")
        right_v.addWidget(self.camera_box)
        cam_layout = QFormLayout()
        cam_container = QWidget()
        cam_container.setLayout(cam_layout)
        self.camera_box.content_layout.addWidget(cam_container)

        # Enable bayer
        self.bayer_chk = QCheckBox("启用拜尔/去马赛克（RGGB）")
        self.bayer_chk.setChecked(getbool("CameraSimulator", "bayer", True))
        cam_layout.addRow(self.bayer_chk)

        # JPEG cycles
        self.jpeg_cycles_spin = QSpinBox()
        self.jpeg_cycles_spin.setRange(0, 10)
        self.jpeg_cycles_spin.setValue(get("CameraSimulator", "jpeg_cycles", 1, int))
        cam_layout.addRow("JPEG压缩次数", self.jpeg_cycles_spin)

        # JPEG quality min/max
        self.jpeg_qmin_spin = QSpinBox()
        self.jpeg_qmin_spin.setRange(1, 100)
        self.jpeg_qmin_spin.setValue(get("CameraSimulator", "jpeg_qmin", 88, int))
        self.jpeg_qmax_spin = QSpinBox()
        self.jpeg_qmax_spin.setRange(1, 100)
        self.jpeg_qmax_spin.setValue(get("CameraSimulator", "jpeg_qmax", 96, int))
        qbox = QHBoxLayout()
        qbox.addWidget(self.jpeg_qmin_spin)
        qbox.addWidget(QLabel("到"))
        qbox.addWidget(self.jpeg_qmax_spin)
        cam_layout.addRow("JPEG质量（最小到最大）", qbox)

        # Vignette strength
        self.vignette_spin = QDoubleSpinBox()
        self.vignette_spin.setRange(0.0, 1.0)
        self.vignette_spin.setSingleStep(0.01)
        self.vignette_spin.setValue(get("CameraSimulator", "vignette_strength", 0.35, float))
        cam_layout.addRow("暗角强度", self.vignette_spin)

        # Chromatic aberration strength
        self.chroma_spin = QDoubleSpinBox()
        self.chroma_spin.setRange(0.0, 10.0)
        self.chroma_spin.setSingleStep(0.1)
        self.chroma_spin.setValue(get("CameraSimulator", "chroma_strength", 1.2, float))
        cam_layout.addRow("色差（像素）", self.chroma_spin)

        # ISO scale
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setRange(0.1, 16.0)
        self.iso_spin.setSingleStep(0.1)
        self.iso_spin.setValue(get("CameraSimulator", "iso_scale", 1.0, float))
        cam_layout.addRow("ISO/曝光缩放", self.iso_spin)

        # Read noise
        self.read_noise_spin = QDoubleSpinBox()
        self.read_noise_spin.setRange(0.0, 50.0)
        self.read_noise_spin.setSingleStep(0.1)
        self.read_noise_spin.setValue(get("CameraSimulator", "read_noise", 2.0, float))
        cam_layout.addRow("读取噪声（DN）", self.read_noise_spin)

        # Hot pixel prob
        self.hot_pixel_spin = QDoubleSpinBox()
        self.hot_pixel_spin.setDecimals(9)
        self.hot_pixel_spin.setRange(0.0, 1.0)
        self.hot_pixel_spin.setSingleStep(1e-6)
        self.hot_pixel_spin.setValue(get("CameraSimulator", "hot_pixel_prob", 1e-6, float))
        cam_layout.addRow("坏点概率", self.hot_pixel_spin)

        # Banding strength
        self.banding_spin = QDoubleSpinBox()
        self.banding_spin.setRange(0.0, 1.0)
        self.banding_spin.setSingleStep(0.01)
        self.banding_spin.setValue(get("CameraSimulator", "banding_strength", 0.0, float))
        cam_layout.addRow("条带强度", self.banding_spin)

        # Motion blur kernel
        self.motion_blur_spin = QSpinBox()
        self.motion_blur_spin.setRange(1, 51)
        self.motion_blur_spin.setValue(get("CameraSimulator", "motion_blur_kernel", 1, int))
        cam_layout.addRow("运动模糊核", self.motion_blur_spin)

        self.camera_box.setVisible(getbool("CameraSimulator", "enabled", False))
        self.params_box.setVisible(not getbool("General", "auto_mode", True))
        self.texture_box.setVisible(not getbool("General", "auto_mode", True))

        self.ref_hint = QLabel("AWB使用'AWB参考'选择器。FFT频谱匹配使用'FFT参考'选择器。")
        right_v.addWidget(self.ref_hint)

        self.analysis_input = AnalysisPanel(title="输入分析")
        self.analysis_output = AnalysisPanel(title="输出分析")
        right_v.addWidget(self.analysis_input)
        right_v.addWidget(self.analysis_output)

        right_v.addStretch(1)

        # Status bar
        self.status = QLabel("就绪")
        self.status.setStyleSheet("color:#bdbdbd;padding:6px")
        self.status.setAlignment(Qt.AlignLeft)
        self.status.setFixedHeight(28)
        self.status.setContentsMargins(6, 6, 6, 6)
        self.statusBar().addWidget(self.status)

        self.worker = None
        self._on_auto_mode_toggled(self.auto_mode_chk.checkState())

    def _on_sim_camera_toggled(self, state):
        enabled = state == Qt.Checked
        self.camera_box.setVisible(enabled)

    def _on_auto_mode_toggled(self, state):
        is_auto = (state == Qt.Checked)
        self.auto_box.setVisible(is_auto)
        self.params_box.setVisible(not is_auto)
        self.texture_box.setVisible(not is_auto)
        self.camera_box.setVisible(not is_auto)
        self.blend_box.setVisible(not is_auto)

    def _update_strength_label(self, value):
        self.strength_label.setText(str(value))

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", str(Path.home()), 
                                            "图片和视频 (*.png *.jpg *.jpeg *.bmp *.tif *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;"
                                            "图片 (*.png *.jpg *.jpeg *.bmp *.tif);;"
                                            "视频 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;"
                                            "所有文件 (*)")
        if path:
            self.input_line.setText(path)
            # Check if it's a video
            ext = Path(path).suffix.lower()
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
            if ext in video_exts:
                self.preview_in.setText("视频文件\n(不支持预览)")
                self.preview_in.setStyleSheet("background:#121213;border:1px solid #2b2b2b;color:#bdbdbd;border-radius:6px")
                self.analysis_input.setEnabled(False)
                # Suggest output as mp4
                out_suggest = str(Path(path).with_name(Path(path).stem + "_out.mp4"))
            else:
                self.load_preview(self.preview_in, path)
                self.analysis_input.update_from_path(path)
                self.analysis_input.setEnabled(True)
                out_suggest = str(Path(path).with_name(Path(path).stem + "_out" + Path(path).suffix))
            if not self.output_line.text():
                self.output_line.setText(out_suggest)

    def choose_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择AWB参考图片", str(Path.home()), "图片 (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.ref_line.setText(path)

    def choose_fft_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择FFT参考图片", str(Path.home()), "图片 (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.fft_ref_line.setText(path)

    def choose_output(self):
        # Check input type to suggest appropriate filter
        input_path = self.input_line.text()
        ext = Path(input_path).suffix.lower() if input_path else ''
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        
        if ext in video_exts:
            filter_str = "MP4 (*.mp4);;AVI (*.avi);;MOV (*.mov);;MKV (*.mkv);;所有文件 (*)"
        else:
            filter_str = "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif);;所有文件 (*)"
            
        path, _ = QFileDialog.getSaveFileName(self, "选择输出路径", str(Path.home()), filter_str)
        if path:
            self.output_line.setText(path)

    def choose_lut(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择LUT文件", str(Path.home()), "LUT文件 (*.png *.npy *.cube);;所有文件 (*)")
        if path:
            self.lut_line.setText(path)

    def _on_lut_toggled(self, state):
        visible = (state == Qt.Checked)
        for w in self._lut_controls:
            w.setVisible(visible)

    def _on_video_memory_toggled(self, state):
        enabled = (state == Qt.Checked)
        self.video_memory_gb_spin.setEnabled(enabled)
        if enabled:
            self.video_buffer_frames_chk.setChecked(False)

    def _on_video_buffer_frames_toggled(self, state):
        enabled = (state == Qt.Checked)
        self.video_buffer_frames_spin.setEnabled(enabled)
        if enabled:
            self.video_memory_gb_chk.setChecked(False)

    def load_preview(self, widget: QLabel, path: str):
        if not path or not os.path.exists(path):
            widget.setText("无图片")
            widget.setPixmap(QPixmap())
            return
        pix = qpixmap_from_path(path, max_size=(widget.width(), widget.height()))
        widget.setPixmap(pix)

    def set_enabled_all(self, enabled: bool):
        for w in self.findChildren((QPushButton, QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, QCheckBox, QSlider, QToolButton)):
            w.setEnabled(enabled)

    def on_run(self):
        from types import SimpleNamespace
        inpath = self.input_line.text().strip()
        outpath = self.output_line.text().strip()
        if not inpath or not os.path.exists(inpath):
            QMessageBox.warning(self, "缺少输入", "请选择有效的输入文件。")
            return
        if not outpath:
            QMessageBox.warning(self, "缺少输出", "请选择输出路径。")
            return

        awb_ref_val = self.ref_line.text() or None
        fft_ref_val = self.fft_ref_line.text() or None
        args = SimpleNamespace()
        args.clahe = False
        args.lbp = False

        if self.auto_mode_chk.isChecked():
            strength = self.strength_slider.value() / 100.0
            args.noise_std = strength * 0.04
            args.cutoff = max(0.01, 0.4 - strength * 0.3)
            args.fstrength = strength * 0.95
            args.phase_perturb = strength * 0.1
            args.perturb = True
            args.perturb_magnitude = strength * 0.015
            args.jpeg_cycles = int(strength * 2)
            args.jpeg_qmin = max(1, int(95 - strength * 35))
            args.jpeg_qmax = max(1, int(99 - strength * 25))
            args.vignette_strength = strength * 0.6
            args.chroma_strength = strength * 2.0
            args.motion_blur_kernel = 1 + 2 * int(strength * 6)
            args.banding_strength = strength * 0.1
            args.randomness = 0.05
            args.radial_smooth = 5
            args.fft_mode = "auto"
            args.fft_alpha = 1.0
            args.alpha = 1.0
            args.fft_variant = self.fft_variant_combo.currentText()
            args.glcm = False
            args.glcm_distances = [1]
            args.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            args.glcm_levels = 256
            args.glcm_strength = 0.9
            seed_val = int(self.seed_spin.value())
            args.seed = None if seed_val == 0 else seed_val
            args.sim_camera = True
            args.no_no_bayer = True
            args.iso_scale = 1.0
            args.read_noise = 2.0
            args.hot_pixel_prob = 1e-6
            args.noise = True
            args.fft = True
            args.blend = True
            args.blend_tolerance = int(math.ceil(10*strength))
            args.blend_min_region = int(math.ceil(-5.1111 / max(strength, 0.1) + 55.1111))
            args.blend_max_samples = int(444444.4444 * min(max(strength, 0.1), 1.0) + 55555.5556)
            args.blend_n_jobs = os.cpu_count() or 4
        else:
            seed_val = int(self.seed_spin.value())
            args.seed = None if seed_val == 0 else seed_val
            sim_camera = bool(self.sim_camera_chk.isChecked())
            enable_bayer = bool(self.bayer_chk.isChecked())
            args.noise_std = float(self.noise_spin.value())
            args.cutoff = float(self.cutoff_spin.value())
            args.fstrength = float(self.fstrength_spin.value())
            args.strength = float(self.fstrength_spin.value())
            args.randomness = float(self.randomness_spin.value())
            args.phase_perturb = float(self.phase_perturb_spin.value())
            args.fft_mode = self.fft_mode_combo.currentText()
            args.fft_alpha = float(self.fft_alpha_spin.value())
            args.alpha = float(self.fft_alpha_spin.value())
            args.radial_smooth = int(self.radial_smooth_spin.value())
            args.sim_camera = sim_camera
            args.no_no_bayer = bool(enable_bayer)
            args.jpeg_cycles = int(self.jpeg_cycles_spin.value())
            args.jpeg_qmin = int(self.jpeg_qmin_spin.value())
            args.jpeg_qmax = int(self.jpeg_qmax_spin.value())
            args.vignette_strength = float(self.vignette_spin.value())
            args.chroma_strength = float(self.chroma_spin.value())
            args.iso_scale = float(self.iso_spin.value())
            args.read_noise = float(self.read_noise_spin.value())
            args.hot_pixel_prob = float(self.hot_pixel_spin.value())
            args.banding_strength = float(self.banding_spin.value())
            args.motion_blur_kernel = int(self.motion_blur_spin.value())
            args.glcm = bool(self.glcm_chk.isChecked())
            args.glcm_distances = [int(x) for x in self.glcm_distances_line.text().split()]
            args.glcm_angles = [float(x) for x in self.glcm_angles_line.text().split()]
            args.glcm_levels = int(self.glcm_levels_spin.value())
            args.glcm_strength = float(self.glcm_strength_spin.value())
            args.noise = self.noise_enable_chk.isChecked()
            args.fft = self.fft_enable_chk.isChecked()
            args.fft_variant = self.fft_variant_combo.currentText()
            args.perturb = self.perturb_enable_chk.isChecked()
            args.perturb_magnitude = float(self.perturb_spin.value())
            args.blend = bool(self.blend_chk.isChecked())
            args.blend_tolerance = int(self.blend_tolerance.value())
            args.blend_min_region = int(self.blend_min_region.value())
            args.blend_max_samples = int(self.blend_max_samples.value())
            args.blend_n_jobs = int(self.blend_n_jobs.value())

        # AI Normalizer
        args.non_semantic = bool(self.ns_chk.isChecked())
        if args.non_semantic:
            try:
                import torch
            except ImportError:
                QMessageBox.warning(self, "缺少依赖", "AI归一化需要Torch (PyTorch)，但尚未安装。")
                self.set_enabled_all(True)
                return
            args.ns_iterations = int(self.ns_iterations_spin.value())
            args.ns_learning_rate = float(self.ns_lr_spin.value())
            args.ns_t_lpips = float(self.ns_t_lpips_spin.value())
            args.ns_t_l2 = float(self.ns_t_l2_spin.value())
            args.ns_c_lpips = float(self.ns_c_lpips_spin.value())
            args.ns_c_l2 = float(self.ns_c_l2_spin.value())
            args.ns_grad_clip = float(self.ns_grad_clip_spin.value())

        # AWB handling
        if self.awb_chk.isChecked():
            args.awb = True
            args.ref = awb_ref_val
        else:
            args.awb = False
            args.ref = None

        # FFT spectral matching reference
        args.fft_ref = fft_ref_val

        # LUT handling
        if self.lut_chk.isChecked():
            lut_path = self.lut_line.text().strip()
            args.lut = lut_path if lut_path else None
            args.lut_strength = float(self.lut_strength_spin.value())
        else:
            args.lut = None
            args.lut_strength = 1.0
        
        # Task count
        args.task_count = self.task_count_spin.value()

        # Video buffer settings
        if self.video_memory_gb_chk.isChecked():
            args.video_memory_gb = float(self.video_memory_gb_spin.value())
        else:
            args.video_memory_gb = None

        if self.video_buffer_frames_chk.isChecked():
            args.video_buffer_frames = int(self.video_buffer_frames_spin.value())
        else:
            args.video_buffer_frames = None

        self.worker = Worker(inpath, outpath, args)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.started.connect(lambda: self.on_worker_started())
        self.worker.start()

        self.progress.setRange(0, 0)
        self.status.setText("处理中...")
        self.set_enabled_all(False)
        
    def on_progress_update(self, current, total):
        """Update progress bar for video processing"""
        if current == -2:
            # 分段并行处理阶段
            self.progress.setRange(0, 0)
            self.status.setText("正在并行处理视频段...")
        elif current == -1:
            # FFmpeg合并阶段
            self.progress.setRange(0, 0)  # 显示忙碌状态
            self.status.setText("正在合并视频...")
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
            self.status.setText(f"处理中... 帧 {current}/{total}")

    def on_worker_started(self):
        pass

    def on_finished(self, outpath):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText("完成 — 已保存到: " + outpath)
        
        # Check if output is video
        ext = Path(outpath).suffix.lower()
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        if ext in video_exts:
            self.preview_out.setText("视频已处理完成\n(不支持预览)")
            self.preview_out.setStyleSheet("background:#121213;border:1px solid #2b2b2b;color:#bdbdbd;border-radius:6px")
            self.analysis_output.setEnabled(False)
        else:
            self.load_preview(self.preview_out, outpath)
            self.analysis_output.update_from_path(outpath)
            self.analysis_output.setEnabled(True)
        self.set_enabled_all(True)

    def on_error(self, msg, traceback_text):
        from PyQt5.QtWidgets import QDialog, QTextEdit
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("错误")

        dialog = QDialog(self)
        dialog.setWindowTitle("处理错误")
        dialog.setMinimumSize(700, 480)
        layout = QVBoxLayout(dialog)

        error_label = QLabel(f"错误: {msg}")
        error_label.setWordWrap(True)
        layout.addWidget(error_label)

        traceback_edit = QTextEdit()
        traceback_edit.setReadOnly(True)
        traceback_edit.setText(traceback_text)
        traceback_edit.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(traceback_edit)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        dialog.exec_()
        self.set_enabled_all(True)

    def open_output_folder(self):
        out = self.output_line.text().strip()
        if not out:
            QMessageBox.information(self, "无输出", "尚未设置输出路径。")
            return
        folder = os.path.dirname(os.path.abspath(out))
        if not os.path.exists(folder):
            QMessageBox.warning(self, "未找到", "输出文件夹不存在: " + folder)
            return
        if sys.platform.startswith('darwin'):
            os.system(f'open "{folder}"')
        elif os.name == 'nt':
            os.startfile(folder)