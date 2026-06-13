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
DEFAULT_CLASSIFY_BW_STD_THRESH = 20.0
DEFAULT_CLASSIFY_EDGE_RATIO_MIN = 0.03
DEFAULT_CLASSIFY_LINE_COUNT_MIN = 3
DEFAULT_AUTO_PERCENTILE_LOW = 5.0
DEFAULT_AUTO_PERCENTILE_HIGH = 95.0
DEFAULT_JPG_QUALITY = 95
DEFAULT_OUTPUT_COLOR_MODE = "auto"  # auto / color / grayscale / binary
DEFAULT_PRINTER_NAME = "priPrinter"


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
        "auto_perspective":  cfg.getboolean("processing", "auto_perspective",  fallback=False),
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
        "autofix_enabled":   str(settings.get("autofix_enabled",   True)).lower(),
        "auto_apply_autofix": str(settings.get("auto_apply_autofix", True)).lower(),
        "hdr_in_autofix":    str(settings.get("hdr_in_autofix",    True)).lower(),
        "auto_perspective":  str(settings.get("auto_perspective", True)).lower(),
        "shadow_highlight_strength": str(settings.get("shadow_highlight_strength", DEFAULT_SHADOW_HIGHLIGHT_STRENGTH)),
        "sharpen_strength":  str(settings.get("sharpen_strength", DEFAULT_SHARPEN_STRENGTH)),
        "hdr_strength":      str(settings.get("hdr_strength",     DEFAULT_HDR_STRENGTH)),
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

    target = _get_path(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        cfg.write(f)
