#!/usr/bin/env python3
"""
video_processor.py

Video processing module using OpenCV and FFmpeg.
Handles reading video frames, processing each frame, and writing back to video.
Supports parallel segmented processing for maximum speed.
"""

import os
import tempfile
import subprocess
import numpy as np
from PIL import Image
from .processor import process_image
from PyQt5.QtCore import QThread, pyqtSignal
import threading
import queue
import concurrent.futures

# 启用OpenCV优化
try:
    import cv2
    cv2.setUseOptimized(True)
    cv2.setNumThreads(os.cpu_count() or 4)
except ImportError:
    pass


def get_ffmpeg_path():
    """Get FFmpeg path, checking imageio-ffmpeg first"""
    # Try imageio-ffmpeg first
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception):
        pass
    
    # Try system FFmpeg
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    return None


def is_video_file(path):
    """Check if a file is a video based on extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return os.path.splitext(path.lower())[1] in video_extensions


def process_video_frame(arr, args, ref_arr_fft=None, ref_arr_awb=None, lut=None):
    """Process a single video frame (numpy array) using existing processing logic"""
    from .utils import (
        add_gaussian_noise,
        clahe_color_correction,
        randomized_perturbation,
        fourier_match_spectrum,
        auto_white_balance_ref,
        apply_lut,
        glcm_normalize,
        lbp_normalize,
        attack_non_semantic,
        blend_colors,
        FOURIER_VARIANTS,
    )
    from .camera_pipeline import simulate_camera_pipeline

    # Blend system
    if args.blend:
        try:
            arr = blend_colors(arr, tolerance=args.blend_tolerance, min_region_size=args.blend_min_region,
                               max_kmeans_samples=args.blend_max_samples, n_jobs=args.blend_n_jobs)
        except Exception as e:
            pass

    # Non-semantic attack
    if args.non_semantic:
        try:
            arr = attack_non_semantic(
                arr,
                iterations=args.ns_iterations,
                learning_rate=args.ns_learning_rate,
                t_lpips=args.ns_t_lpips,
                t_l2=args.ns_t_l2,
                c_lpips=args.ns_c_lpips,
                c_l2=args.ns_c_l2,
                grad_clip_value=args.ns_grad_clip
            )
        except Exception as e:
            pass

    # CLAHE
    if args.clahe:
        arr = clahe_color_correction(arr, clip_limit=args.clahe_clip, tile_grid_size=(args.tile, args.tile))

    # FFT
    if args.fft:
        fft_variant = getattr(args, 'fft_variant', 'v2')
        fft_func = FOURIER_VARIANTS.get(fft_variant, fourier_match_spectrum)
        fft_kwargs = dict(ref_img_arr=ref_arr_fft, mode=args.fft_mode,
                          alpha=args.fft_alpha, cutoff=args.cutoff,
                          strength=args.fstrength, randomness=args.randomness,
                          seed=args.seed)
        if fft_variant != 'v3':
            fft_kwargs['phase_perturb'] = args.phase_perturb
        fft_kwargs['radial_smooth'] = args.radial_smooth
        arr = fft_func(arr, **fft_kwargs)

    # GLCM
    if args.glcm:
        arr = glcm_normalize(arr, ref_img_arr=ref_arr_fft, distances=args.glcm_distances,
                            angles=args.glcm_angles, levels=args.glcm_levels,
                            strength=args.glcm_strength, seed=args.seed)

    # LBP
    if args.lbp:
        arr = lbp_normalize(arr, ref_img_arr=ref_arr_fft, radius=args.lbp_radius,
                             n_points=args.lbp_n_points, method=args.lbp_method,
                             strength=args.lbp_strength, seed=args.seed)

    # Noise
    if args.noise:
        arr = add_gaussian_noise(arr, std_frac=args.noise_std, seed=args.seed)

    # Perturbation
    if args.perturb:
        arr = randomized_perturbation(arr, magnitude_frac=args.perturb_magnitude, seed=args.seed)

    # Camera pipeline
    if args.sim_camera:
        arr = simulate_camera_pipeline(arr,
                                       bayer=not args.no_no_bayer,
                                       jpeg_cycles=args.jpeg_cycles,
                                       jpeg_quality_range=(args.jpeg_qmin, args.jpeg_qmax),
                                       vignette_strength=args.vignette_strength,
                                       chroma_aberr_strength=args.chroma_strength,
                                       iso_scale=args.iso_scale,
                                       read_noise_std=args.read_noise,
                                       hot_pixel_prob=args.hot_pixel_prob,
                                       banding_strength=args.banding_strength,
                                       motion_blur_kernel=args.motion_blur_kernel,
                                       seed=args.seed)

    # AWB
    if args.awb:
        if args.ref and ref_arr_awb is not None:
            try:
                arr = auto_white_balance_ref(arr, ref_arr_awb)
            except Exception as e:
                pass
        else:
            arr = auto_white_balance_ref(arr, None)

    # LUT
    if args.lut and lut is not None:
        try:
            arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
            arr_lut = apply_lut(arr_uint8, lut, strength=args.lut_strength)
            arr = np.clip(arr_lut, 0, 255).astype(np.uint8)
        except Exception as e:
            pass

    return np.clip(arr, 0, 255).astype(np.uint8)


def process_video(inpath, outpath, args, progress_callback=None):
    """Process video with optimized performance - 使用FFmpeg管道零IO开销"""
    return process_video_optimized(inpath, outpath, args, progress_callback)


def _process_video_single_threaded(cap, outpath, args, ref_arr_fft, ref_arr_awb, lut, fps, width, height, total_frames, progress_callback):
    """Single-threaded processing as fallback"""
    import cv2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = outpath + ".temp.mp4"
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    frame_count = 0
    update_interval = max(1, min(10, total_frames // 100))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_rgb = process_video_frame(frame_rgb, args, ref_arr_fft, ref_arr_awb, lut)
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        out.write(processed_bgr)
        
        frame_count += 1
        if progress_callback and (frame_count % update_interval == 0 or frame_count == total_frames):
            progress_callback(frame_count, total_frames)
    
    cap.release()
    out.release()
    
    if progress_callback:
        progress_callback(-1, total_frames)
    
    _combine_video_with_audio(temp_video, inpath, outpath)


def _process_video_single_threaded_fallback(inpath, outpath, args, ref_arr_fft, ref_arr_awb, lut, progress_callback):
    """真正的回退函数，不依赖 process_video"""
    import cv2
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {inpath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return _process_video_single_threaded(cap, outpath, args, ref_arr_fft, ref_arr_awb, lut, fps, width, height, total_frames, progress_callback)


def _concat_segments_with_audio(concat_file, original_video, output_video):
    """Concatenate segments and add audio"""
    ffmpeg_path = get_ffmpeg_path()
    
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg or install imageio-ffmpeg.")
    
    cmd = [
        ffmpeg_path,
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-i', original_video,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest',
        output_video
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")


def _combine_video_with_audio(video_no_audio, original_video, output_video):
    """Combine processed video with original audio - 使用快速编码预设"""
    ffmpeg_path = get_ffmpeg_path()
    
    if ffmpeg_path:
        cmd = [
            ffmpeg_path,
            '-y',
            '-i', video_no_audio,
            '-i', original_video,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_video
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Fallback: just rename the video without audio
            import shutil
            shutil.move(video_no_audio, output_video)
        finally:
            if os.path.exists(video_no_audio):
                try:
                    os.remove(video_no_audio)
                except:
                    pass
    else:
        # No FFmpeg: just rename the video without audio
        import shutil
        shutil.move(video_no_audio, output_video)


def _combine_frames_with_ffmpeg_jpg(frames_dir, original_video, output_video, fps):
    """Combine processed JPG frames with original audio using FFmpeg - 快速编码"""
    ffmpeg_path = get_ffmpeg_path()
    
    if ffmpeg_path:
        frame_pattern = os.path.join(frames_dir, "frame_%08d.jpg")
        
        cmd = [
            ffmpeg_path,
            '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-i', original_video,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_video
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Fallback to OpenCV-only
            _combine_frames_with_opencv(frames_dir, output_video, fps, 'jpg')
    else:
        # No FFmpeg: use OpenCV-only
        _combine_frames_with_opencv(frames_dir, output_video, fps, 'jpg')


def _combine_frames_with_ffmpeg(frames_dir, original_video, output_video, fps):
    """Combine processed frames with original audio using FFmpeg"""
    ffmpeg_path = get_ffmpeg_path()
    
    if ffmpeg_path:
        frame_pattern = os.path.join(frames_dir, "frame_%08d.png")
        
        cmd = [
            ffmpeg_path,
            '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-i', original_video,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            output_video
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Fallback to OpenCV-only
            _combine_frames_with_opencv(frames_dir, output_video, fps, 'png')
    else:
        # No FFmpeg: use OpenCV-only
        _combine_frames_with_opencv(frames_dir, output_video, fps, 'png')


def _combine_frames_with_opencv(frames_dir, output_video, fps, ext='jpg'):
    """Combine frames into video using only OpenCV (no audio)"""
    import cv2
    import glob
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, f"frame_*.{ext}")))
    if not frame_files:
        raise RuntimeError("No frames found to combine")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        raise RuntimeError("Cannot read first frame")
    
    height, width = first_frame.shape[:2]
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write all frames
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
    
    out.release()


def _process_video_with_temp_files(inpath, outpath, args, progress_callback=None):
    """备用方案：使用临时文件处理视频"""
    import cv2
    from .utils import load_lut
    
    ref_arr_fft = None
    if args.fft_ref:
        try:
            ref_img_fft = Image.open(args.fft_ref).convert('RGB')
            ref_arr_fft = np.array(ref_img_fft)
        except Exception as e:
            pass
    
    ref_arr_awb = None
    if args.awb and args.ref:
        try:
            ref_img_awb = Image.open(args.ref).convert('RGB')
            ref_arr_awb = np.array(ref_img_awb)
        except Exception as e:
            pass
    
    # Load LUT once (if needed)
    lut = None
    if args.lut:
        try:
            lut = load_lut(args.lut)
        except Exception as e:
            pass
    
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {inpath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        frame_count = 0
        update_interval = max(1, min(10, total_frames // 100))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_rgb = process_video_frame(frame_rgb, args, ref_arr_fft, ref_arr_awb, lut)
            processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            
            frame_path = os.path.join(processed_frames_dir, f"frame_{frame_count:08d}.jpg")
            cv2.imwrite(frame_path, processed_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            frame_count += 1
            if progress_callback and (frame_count % update_interval == 0 or frame_count == total_frames):
                progress_callback(frame_count, total_frames)
        
        cap.release()
        
        if progress_callback:
            progress_callback(-1, total_frames)
        
        _combine_frames_with_ffmpeg_jpg(processed_frames_dir, inpath, outpath, fps)
        
        if progress_callback:
            progress_callback(total_frames, total_frames)


def process_video_optimized(inpath, outpath, args, progress_callback=None):
    """
    优化版视频处理：使用生产者-消费者模式 + 内存滑动窗口 + FFmpeg管道
    零IO开销 + 防内存爆炸
    """
    import cv2
    from .utils import load_lut
    
    # 检查FFmpeg是否可用
    ffmpeg_path = get_ffmpeg_path()
    
    if not ffmpeg_path:
        # No FFmpeg, use single-threaded fallback with OpenCV only
        return _process_video_single_threaded_fallback(inpath, outpath, args, progress_callback)
    
    # 加载引用图片（一次）
    ref_arr_fft = None
    if args.fft_ref:
        try:
            ref_img_fft = Image.open(args.fft_ref).convert('RGB')
            ref_arr_fft = np.array(ref_img_fft)
        except Exception as e:
            pass
    
    ref_arr_awb = None
    if args.awb and args.ref:
        try:
            ref_img_awb = Image.open(args.ref).convert('RGB')
            ref_arr_awb = np.array(ref_img_awb)
        except Exception as e:
            pass
    
    # 加载LUT（一次）
    lut = None
    if args.lut:
        try:
            lut = load_lut(args.lut)
        except Exception as e:
            pass
    
    # 打开视频获取信息
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {inpath}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 滑动窗口大小 - 限制内存中最多同时存放的帧数
    # 优先级：1. 内存用量 2. 帧数 3. 自动
    if hasattr(args, 'video_memory_gb') and args.video_memory_gb is not None:
        # 用户直接设置内存用量 (GB)
        target_memory_gb = float(args.video_memory_gb)
        frame_size_mb = (width * height * 3) / (1024 * 1024)
        window_size = int((target_memory_gb * 1024) / max(frame_size_mb, 1))
        window_size = min(1024, max(16, window_size))  # 16-1024帧
    elif hasattr(args, 'video_buffer_frames') and args.video_buffer_frames is not None:
        # 用户设置帧数
        window_size = int(args.video_buffer_frames)
        window_size = min(1024, max(16, window_size))
    else:
        # 自动调整
        frame_size_mb = (width * height * 3) / (1024 * 1024)
        
        # 目标内存占用：最多 2GB 或 可用内存的 1/4，取较小值
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            target_memory_gb = min(2.0, available_gb / 4)
        except (ImportError, Exception):
            # 如果无法获取内存信息，默认目标 1GB
            target_memory_gb = 1.0
        
        # 计算窗口大小，最小 32 帧，最大 512 帧
        window_size = int((target_memory_gb * 1024) / max(frame_size_mb, 1))
        window_size = min(512, max(32, window_size))
    
    # 创建队列 - 有限大小，防止内存爆炸
    frame_queue = queue.Queue(maxsize=window_size)
    stop_event = threading.Event()
    
    # 使用条件变量和滑动窗口控制
    result_dict = {}
    result_lock = threading.Lock()
    result_cond = threading.Condition(result_lock)
    
    # 写入进度
    written_count = 0
    written_lock = threading.Lock()
    
    # 确定工作线程数
    worker_count = getattr(args, 'task_count', os.cpu_count() or 4)
    if worker_count <= 0:
        worker_count = os.cpu_count() or 4
    
    if progress_callback:
        progress_callback(-2, total_frames)
    
    try:
        # === 生产者线程：读取视频帧 ===
        def producer():
            try:
                cap_prod = cv2.VideoCapture(inpath)
                frame_idx = 0
                while not stop_event.is_set() and frame_idx < total_frames:
                    ret, frame = cap_prod.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 检查是否超过滑动窗口
                    with written_lock:
                        current_written = written_count
                    
                    # 如果当前生产的帧索引超过 写入进度 + 窗口大小，则阻塞
                    while frame_idx >= current_written + window_size and not stop_event.is_set():
                        import time
                        time.sleep(0.01)
                        with written_lock:
                            current_written = written_count
                    
                    if stop_event.is_set():
                        break
                    
                    frame_queue.put((frame_idx, frame_rgb))
                    frame_idx += 1
                
                cap_prod.release()
            except Exception as e:
                pass
            finally:
                # 发送结束信号
                for _ in range(worker_count):
                    frame_queue.put(None)
        
        # === 消费者线程：处理帧 ===
        def worker():
            while not stop_event.is_set():
                item = frame_queue.get()
                if item is None:
                    frame_queue.task_done()
                    break
                
                try:
                    frame_idx, frame_rgb = item
                    processed_rgb = process_video_frame(frame_rgb, args, ref_arr_fft, ref_arr_awb, lut)
                    processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                    
                    with result_cond:
                        result_dict[frame_idx] = processed_bgr
                        result_cond.notify_all()
                except Exception as e:
                    pass
                finally:
                    frame_queue.task_done()
        
        # === 写入线程：按顺序写入FFmpeg管道 ===
        def writer():
            nonlocal written_count
            try:
                # 构建FFmpeg命令 - 直接管道输入，避免临时文件
                cmd = [
                    ffmpeg_path,
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', f'{width}x{height}',
                    '-r', str(fps),
                    '-i', '-',
                    '-i', inpath,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    outpath
                ]
                
                # 启动FFmpeg进程
                ffmpeg_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                local_written = 0
                while local_written < total_frames and not stop_event.is_set():
                    # 等待当前需要的帧处理完成
                    with result_cond:
                        while local_written not in result_dict and not stop_event.is_set():
                            result_cond.wait(timeout=0.1)
                        
                        if stop_event.is_set():
                            break
                        
                        # 写入帧
                        frame = result_dict.pop(local_written)
                    
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        break
                    
                    local_written += 1
                    with written_lock:
                        written_count = local_written
                    
                    # 更新进度
                    if progress_callback and (local_written % 10 == 0 or local_written == total_frames):
                        progress_callback(local_written, total_frames)
                
                # 关闭管道并等待FFmpeg完成
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
                
            except Exception as e:
                pass
        
        # === 启动所有线程 ===
        threads = []
        
        # 生产者
        producer_thread = threading.Thread(target=producer, daemon=True)
        threads.append(producer_thread)
        
        # 工作者
        for _ in range(worker_count):
            worker_thread = threading.Thread(target=worker, daemon=True)
            threads.append(worker_thread)
        
        # 写入者
        writer_thread = threading.Thread(target=writer, daemon=True)
        threads.append(writer_thread)
        
        # 启动
        for t in threads:
            t.start()
        
        # 等待完成
        producer_thread.join()
        for t in threads[1:-1]:  # 等待worker
            t.join()
        writer_thread.join()
        
        if progress_callback:
            progress_callback(-1, total_frames)
            progress_callback(total_frames, total_frames)
            
    except Exception as e:
        stop_event.set()
        # 回退到原来的方法
        return _process_video_single_threaded_fallback(inpath, outpath, args, ref_arr_fft, ref_arr_awb, lut, progress_callback)
