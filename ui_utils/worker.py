#!/usr/bin/env python3
"""
Worker thread for image and video processing.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import traceback

try:
    from image_postprocess import process_image, process_video, is_video_file
except Exception:
    process_image = None
    process_video = None
    is_video_file = None
    IMPORT_ERROR = "Could not import process_image module"
else:
    IMPORT_ERROR = None

class Worker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)

    def __init__(self, inpath, outpath, args):
        super().__init__()
        self.inpath = inpath
        self.outpath = outpath
        self.args = args

    def run(self):
        try:
            if process_image is None or (process_video is None and is_video_file is None):
                raise RuntimeError("Could not import processing module: " + (IMPORT_ERROR or "unknown"))
            
            if is_video_file and is_video_file(self.inpath):
                process_video(self.inpath, self.outpath, self.args, 
                            progress_callback=lambda curr, total: self.progress.emit(curr, total))
            else:
                process_image(self.inpath, self.outpath, self.args)
            self.finished.emit(self.outpath)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(str(e), tb)