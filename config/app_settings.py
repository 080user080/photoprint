"""
Читання та запис settings.ini.
Не залежить від жодного іншого модуля проєкту.
"""

import configparser
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "settings.ini")

# Константи для default налаштувань
DEFAULT_WINDOW_WIDTH = 1100
DEFAULT_WINDOW_HEIGHT = 680
DEFAULT_QUEUE_WIDTH = 200
DEFAULT_SHARPEN_STRENGTH = 0.4
DEFAULT_HDR_STRENGTH = 0.5
DEFAULT_SHADOW_HIGHLIGHT_STRENGTH = 0.0
DEFAULT_AUTOSHARP_THRESHOLD = 80.0
DEFAULT_AUTOSHARP_MAX_STRENGTH = 0.7
DEFAULT_CLASSIFY_BW_STD_THRESH = 10.0
DEFAULT_CLASSIFY_EDGE_RATIO_MIN = 0.03
DEFAULT_CLASSIFY_LINE_COUNT_MIN = 3
DEFAULT_AUTO_PERCENTILE_LOW = 5.0
DEFAULT_AUTO_PERCENTILE_HIGH = 95.0
DEFAULT_JPG_QUALITY = 95
DEFAULT_OUTPUT_COLOR_MODE = "auto"  # auto / color / grayscale / binary
DEFAULT_PRINTER_NAME = "priPrinter"
DEFAULT_PARTIAL_PERSPECTIVE = False  # PRIO 2: часткова перспектива
DEFAULT_MINIMIZE_TO_TRAY = False      # PRIO 9: трей-режим
DEFAULT_AUTOFIX_CONTRAST = 0.15
DEFAULT_CONTRAST_MODE = "linear"   # "linear", "percentile", "s_curve", "adaptive"

# Константи для тестових зображень
DEFAULT_TEST_IMAGES_ENABLED = False
DEFAULT_TEST_IMAGES_FOLDER = ""   # порожньо = <корінь проєкту>/tests/test_images

# Константи для видалення тіней
DEFAULT_SHADOW_REMOVE_ENABLED = True
DEFAULT_SHADOW_DETECT_THRESHOLD = 80.0
DEFAULT_SHADOW_DETECT_RATIO = 0.3
DEFAULT_SHADOW_COARSE_BLEND_COLOR = 0.0
DEFAULT_SHADOW_BGR_MODE = False
DEFAULT_SHADOW_UNIFORMITY_LOW = 0.30    # нижній поріг однорідності фону
DEFAULT_SHADOW_UNIFORMITY_HIGH = 0.55   # верхній поріг однорідності фону
DEFAULT_SHADOW_UNIFORMITY_PHOTO_HIGH = 0.65  # верхній поріг для фото (суворіший)
DEFAULT_SHADOW_REMOVE_MODE = "auto"           # auto / always / never
DEFAULT_SHADOW_UNIFORM_STD_THRESHOLD = 2.0   # поріг std блоків для детекції тіні на рівномірному фоні
DEFAULT_SHADOW_UNIFORM_BLOCK_SIZE = 32        # розмір блоку для локального std

# Константи для паралельної обробки
import os as _os

DEFAULT_WORKER_THREADS = min(_os.cpu_count() or 4, 8)
# Для Ryzen 7 7700: cpu_count()=16, обмежуємо 8 щоб не вичерпати RAM
# (кожен потік тримає в пам'яті 1 повноформатне зображення ~20-50MB)
DEFAULT_MAX_WORKER_THREADS = 16
DEFAULT_MIN_WORKER_THREADS = 1

# Константи для пресетів стратегій
DEFAULT_PIPELINE_PRESET = "doc_bw"
DEFAULT_PIPELINE_STEPS_ENABLED = "perspective,shadow_remove,brightness,contrast,hdr,sharpen,grayscale,white_background"

# Константи для вигляду
DEFAULT_BACKGROUND_COLOR = "#E8E8E8"
DEFAULT_PREVIEW_BG_COLOR = "#E8E8E8"
DEFAULT_PREVIEW_TEXT_COLOR = "#333333"
DEFAULT_QUEUE_BG_COLOR = "#FFFFFF"
DEFAULT_QUEUE_TEXT_COLOR = "#111111"
DEFAULT_AUTO_CONTRAST_TEXT = True


def _get_path(path=None):
    return path or _DEFAULT_PATH


def load(path=None) -> dict:
    """Повертає словник з усіма налаштуваннями."""
    cfg = configparser.ConfigParser()
    cfg.read(_get_path(path), encoding="utf-8")

    return {
        "default_mode":      cfg.get("general",    "default_mode",    fallback="auto"),
        "window_width":      cfg.getint("general",   "window_width",    fallback=DEFAULT_WINDOW_WIDTH),
        "window_height":     cfg.getint("general",   "window_height",   fallback=DEFAULT_WINDOW_HEIGHT),
        "queue_width":       cfg.getint("general",   "queue_width",     fallback=DEFAULT_QUEUE_WIDTH),
        "autofix_enabled":   cfg.getboolean("processing", "autofix_enabled",   fallback=True),
        "auto_apply_autofix": cfg.getboolean("processing", "auto_apply_autofix", fallback=True),
        "hdr_in_autofix":    cfg.getboolean("processing", "hdr_in_autofix",    fallback=True),
        "auto_brightness_enabled": cfg.getboolean("processing", "auto_brightness_enabled", fallback=False),
        "auto_perspective":  cfg.getboolean("processing", "auto_perspective",  fallback=False),
        "partial_perspective": cfg.getboolean("processing", "partial_perspective", fallback=DEFAULT_PARTIAL_PERSPECTIVE),
        "shadow_highlight_strength": cfg.getfloat("processing", "shadow_highlight_strength", fallback=DEFAULT_SHADOW_HIGHLIGHT_STRENGTH),
        "sharpen_strength":  cfg.getfloat("processing",  "sharpen_strength",   fallback=DEFAULT_SHARPEN_STRENGTH),
        "hdr_strength":      cfg.getfloat("processing",  "hdr_strength",       fallback=DEFAULT_HDR_STRENGTH),
        # Авто-різкість
        "autosharp_threshold":    cfg.getfloat("processing", "autosharp_threshold",    fallback=DEFAULT_AUTOSHARP_THRESHOLD),
        "autosharp_max_strength": cfg.getfloat("processing", "autosharp_max_strength", fallback=DEFAULT_AUTOSHARP_MAX_STRENGTH),
        # Класифікація документів
        "classify_bw_std_thresh":   cfg.getfloat("processing", "classify_bw_std_thresh",   fallback=DEFAULT_CLASSIFY_BW_STD_THRESH),
        "classify_edge_ratio_min":  cfg.getfloat("processing", "classify_edge_ratio_min",  fallback=DEFAULT_CLASSIFY_EDGE_RATIO_MIN),
        "classify_line_count_min":  cfg.getint("processing",   "classify_line_count_min",  fallback=DEFAULT_CLASSIFY_LINE_COUNT_MIN),
        # Процентилі авто-яскравості/контрасту
        "auto_percentile_low":  cfg.getfloat("processing", "auto_percentile_low",  fallback=DEFAULT_AUTO_PERCENTILE_LOW),
        "auto_percentile_high": cfg.getfloat("processing", "auto_percentile_high", fallback=DEFAULT_AUTO_PERCENTILE_HIGH),
        # Бінаризація чб документів
        "bw_binary":         cfg.getboolean("processing", "bw_binary",           fallback=False),
        "output_color_mode": cfg.get("output",           "output_color_mode",  fallback=DEFAULT_OUTPUT_COLOR_MODE),
        "jpg_quality":       cfg.getint("output",        "jpg_quality",         fallback=DEFAULT_JPG_QUALITY),
        "save_folder":       cfg.get("output",           "save_folder",         fallback=""),
        "printer_name":      cfg.get("printer",          "printer_name",        fallback=DEFAULT_PRINTER_NAME),
        "autofix_contrast":  cfg.getfloat("processing",  "autofix_contrast",    fallback=DEFAULT_AUTOFIX_CONTRAST),
        "minimize_to_tray":  cfg.getboolean("general",   "minimize_to_tray",    fallback=DEFAULT_MINIMIZE_TO_TRAY),
        "contrast_mode":     cfg.get("processing",       "contrast_mode",       fallback=DEFAULT_CONTRAST_MODE),
        # Видалення тіней
        "shadow_remove_enabled":     cfg.getboolean("processing", "shadow_remove_enabled",     fallback=DEFAULT_SHADOW_REMOVE_ENABLED),
        "shadow_detect_threshold":   cfg.getfloat("processing",   "shadow_detect_threshold",   fallback=DEFAULT_SHADOW_DETECT_THRESHOLD),
        "shadow_detect_ratio":       cfg.getfloat("processing",   "shadow_detect_ratio",       fallback=DEFAULT_SHADOW_DETECT_RATIO),
        "shadow_coarse_blend_color": cfg.getfloat("processing",   "shadow_coarse_blend_color", fallback=DEFAULT_SHADOW_COARSE_BLEND_COLOR),
        "shadow_bgr_mode":           cfg.getboolean("processing", "shadow_bgr_mode",           fallback=DEFAULT_SHADOW_BGR_MODE),
        "shadow_uniformity_low":     cfg.getfloat("processing",   "shadow_uniformity_low",     fallback=DEFAULT_SHADOW_UNIFORMITY_LOW),
        "shadow_uniformity_high":    cfg.getfloat("processing",   "shadow_uniformity_high",    fallback=DEFAULT_SHADOW_UNIFORMITY_HIGH),
        "shadow_uniformity_photo_high": cfg.getfloat("processing", "shadow_uniformity_photo_high", fallback=DEFAULT_SHADOW_UNIFORMITY_PHOTO_HIGH),
        "shadow_remove_mode":            cfg.get("processing",    "shadow_remove_mode",            fallback=DEFAULT_SHADOW_REMOVE_MODE),
        "shadow_uniform_std_threshold": cfg.getfloat("processing", "shadow_uniform_std_threshold", fallback=DEFAULT_SHADOW_UNIFORM_STD_THRESHOLD),
        "shadow_uniform_block_size":     cfg.getint("processing", "shadow_uniform_block_size",     fallback=DEFAULT_SHADOW_UNIFORM_BLOCK_SIZE),
        "worker_threads": cfg.getint(
            "processing", "worker_threads",
            fallback=DEFAULT_WORKER_THREADS
        ),
        # Пресети стратегій
        "pipeline_preset":        cfg.get("processing", "pipeline_preset",        fallback=DEFAULT_PIPELINE_PRESET),
        "pipeline_steps_enabled": cfg.get("processing", "pipeline_steps_enabled", fallback=DEFAULT_PIPELINE_STEPS_ENABLED),
        # Вигляд
        "background_color": cfg.get("appearance", "background_color", fallback=DEFAULT_BACKGROUND_COLOR),
        "preview_bg_color": cfg.get("appearance", "preview_bg_color", fallback=DEFAULT_PREVIEW_BG_COLOR),
        "preview_text_color": cfg.get("appearance", "preview_text_color", fallback=DEFAULT_PREVIEW_TEXT_COLOR),
        "queue_bg_color": cfg.get("appearance", "queue_bg_color", fallback=DEFAULT_QUEUE_BG_COLOR),
        "queue_text_color": cfg.get("appearance", "queue_text_color", fallback=DEFAULT_QUEUE_TEXT_COLOR),
        "auto_contrast_text": cfg.getboolean("appearance", "auto_contrast_text", fallback=DEFAULT_AUTO_CONTRAST_TEXT),
        # Тестові зображення
        "test_images_enabled": cfg.getboolean("tests", "test_images_enabled", fallback=DEFAULT_TEST_IMAGES_ENABLED),
        "test_images_folder": cfg.get("tests", "test_images_folder", fallback=DEFAULT_TEST_IMAGES_FOLDER),
    }


def save(settings: dict, path=None):
    """Зберігає словник налаштувань у файл .ini."""
    cfg = configparser.ConfigParser()
    cfg["general"] = {
        "default_mode": settings.get("default_mode", "auto"),
        "window_width": str(settings.get("window_width", DEFAULT_WINDOW_WIDTH)),
        "window_height": str(settings.get("window_height", DEFAULT_WINDOW_HEIGHT)),
        "queue_width": str(settings.get("queue_width", DEFAULT_QUEUE_WIDTH)),
    }
    cfg["processing"] = {
        "autofix_enabled":     str(settings.get("autofix_enabled",   True)).lower(),
        "auto_apply_autofix":  str(settings.get("auto_apply_autofix", True)).lower(),
        "hdr_in_autofix":      str(settings.get("hdr_in_autofix",    True)).lower(),
        "auto_brightness_enabled": str(settings.get("auto_brightness_enabled", False)).lower(),
        "auto_perspective":    str(settings.get("auto_perspective", False)).lower(),
        "partial_perspective": str(settings.get("partial_perspective", DEFAULT_PARTIAL_PERSPECTIVE)).lower(),
        "shadow_highlight_strength": str(settings.get("shadow_highlight_strength", DEFAULT_SHADOW_HIGHLIGHT_STRENGTH)),
        "sharpen_strength":    str(settings.get("sharpen_strength", DEFAULT_SHARPEN_STRENGTH)),
        "hdr_strength":        str(settings.get("hdr_strength",     DEFAULT_HDR_STRENGTH)),
        # Авто-різкість
        "autosharp_threshold":    str(settings.get("autosharp_threshold",    DEFAULT_AUTOSHARP_THRESHOLD)),
        "autosharp_max_strength": str(settings.get("autosharp_max_strength", DEFAULT_AUTOSHARP_MAX_STRENGTH)),
        # Класифікація документів
        "classify_bw_std_thresh":   str(settings.get("classify_bw_std_thresh",   DEFAULT_CLASSIFY_BW_STD_THRESH)),
        "classify_edge_ratio_min":  str(settings.get("classify_edge_ratio_min",  DEFAULT_CLASSIFY_EDGE_RATIO_MIN)),
        "classify_line_count_min":  str(settings.get("classify_line_count_min",  DEFAULT_CLASSIFY_LINE_COUNT_MIN)),
        # Процентилі
        "auto_percentile_low":  str(settings.get("auto_percentile_low",  DEFAULT_AUTO_PERCENTILE_LOW)),
        "auto_percentile_high": str(settings.get("auto_percentile_high", DEFAULT_AUTO_PERCENTILE_HIGH)),
        # Бінаризація чб
        "bw_binary": str(settings.get("bw_binary", False)).lower(),
    }
    cfg["output"] = {
        "output_color_mode": settings.get("output_color_mode", DEFAULT_OUTPUT_COLOR_MODE),
        "jpg_quality":      str(settings.get("jpg_quality", DEFAULT_JPG_QUALITY)),
        "save_folder":      settings.get("save_folder", ""),
    }
    cfg["printer"] = {
        "printer_name": settings.get("printer_name", DEFAULT_PRINTER_NAME),
    }
    cfg["processing"]["autofix_contrast"] = str(settings.get("autofix_contrast", DEFAULT_AUTOFIX_CONTRAST))
    cfg["processing"]["contrast_mode"] = settings.get("contrast_mode", DEFAULT_CONTRAST_MODE)
    cfg["processing"]["shadow_remove_enabled"] = str(settings.get("shadow_remove_enabled", DEFAULT_SHADOW_REMOVE_ENABLED)).lower()
    cfg["processing"]["shadow_detect_threshold"] = str(settings.get("shadow_detect_threshold", DEFAULT_SHADOW_DETECT_THRESHOLD))
    cfg["processing"]["shadow_detect_ratio"] = str(settings.get("shadow_detect_ratio", DEFAULT_SHADOW_DETECT_RATIO))
    cfg["processing"]["shadow_coarse_blend_color"] = str(settings.get("shadow_coarse_blend_color", DEFAULT_SHADOW_COARSE_BLEND_COLOR))
    cfg["processing"]["shadow_bgr_mode"] = str(
        settings.get("shadow_bgr_mode", DEFAULT_SHADOW_BGR_MODE)
    ).lower()
    cfg["processing"]["shadow_uniformity_low"] = str(settings.get("shadow_uniformity_low", DEFAULT_SHADOW_UNIFORMITY_LOW))
    cfg["processing"]["shadow_uniformity_high"] = str(settings.get("shadow_uniformity_high", DEFAULT_SHADOW_UNIFORMITY_HIGH))
    cfg["processing"]["shadow_uniformity_photo_high"] = str(settings.get("shadow_uniformity_photo_high", DEFAULT_SHADOW_UNIFORMITY_PHOTO_HIGH))
    cfg["processing"]["shadow_remove_mode"] = settings.get("shadow_remove_mode", DEFAULT_SHADOW_REMOVE_MODE)
    cfg["processing"]["shadow_uniform_std_threshold"] = str(settings.get("shadow_uniform_std_threshold", DEFAULT_SHADOW_UNIFORM_STD_THRESHOLD))
    cfg["processing"]["shadow_uniform_block_size"] = str(settings.get("shadow_uniform_block_size", DEFAULT_SHADOW_UNIFORM_BLOCK_SIZE))
    cfg["general"]["minimize_to_tray"] = str(settings.get("minimize_to_tray", DEFAULT_MINIMIZE_TO_TRAY)).lower()
    cfg["processing"]["worker_threads"] = str(
        settings.get("worker_threads", DEFAULT_WORKER_THREADS)
    )
    cfg["processing"]["pipeline_preset"]        = settings.get("pipeline_preset",        DEFAULT_PIPELINE_PRESET)
    cfg["processing"]["pipeline_steps_enabled"] = settings.get("pipeline_steps_enabled", DEFAULT_PIPELINE_STEPS_ENABLED)
    # Вигляд
    cfg["appearance"] = {
        "background_color": settings.get("background_color", DEFAULT_BACKGROUND_COLOR),
        "preview_bg_color": settings.get("preview_bg_color", DEFAULT_PREVIEW_BG_COLOR),
        "preview_text_color": settings.get("preview_text_color", DEFAULT_PREVIEW_TEXT_COLOR),
        "queue_bg_color": settings.get("queue_bg_color", DEFAULT_QUEUE_BG_COLOR),
        "queue_text_color": settings.get("queue_text_color", DEFAULT_QUEUE_TEXT_COLOR),
        "auto_contrast_text": str(settings.get("auto_contrast_text", DEFAULT_AUTO_CONTRAST_TEXT)).lower(),
    }
    cfg["tests"] = {
        "test_images_enabled": str(settings.get("test_images_enabled", DEFAULT_TEST_IMAGES_ENABLED)).lower(),
        "test_images_folder": settings.get("test_images_folder", DEFAULT_TEST_IMAGES_FOLDER),
    }

    target = _get_path(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        cfg.write(f)