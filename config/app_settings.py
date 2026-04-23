"""
Читання та запис settings.ini.
Не залежить від жодного іншого модуля проєкту.
"""

import configparser
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "settings.ini")


def _get_path(path=None):
    return path or _DEFAULT_PATH


def load(path=None) -> dict:
    """Повертає словник з усіма налаштуваннями."""
    cfg = configparser.ConfigParser()
    cfg.read(_get_path(path), encoding="utf-8")

    return {
        "default_mode":      cfg.get("general",    "default_mode",    fallback="auto"),
        "autofix_enabled":   cfg.getboolean("processing", "autofix_enabled",   fallback=True),
        "hdr_in_autofix":    cfg.getboolean("processing", "hdr_in_autofix",    fallback=True),
        "auto_perspective":  cfg.getboolean("processing", "auto_perspective",  fallback=True),
        "sharpen_strength":  cfg.getfloat("processing",  "sharpen_strength",   fallback=0.4),
        "hdr_strength":      cfg.getfloat("processing",  "hdr_strength",       fallback=0.5),
        # Авто-різкість
        "autosharp_threshold":    cfg.getfloat("processing", "autosharp_threshold",    fallback=80.0),
        "autosharp_max_strength": cfg.getfloat("processing", "autosharp_max_strength", fallback=0.7),
        # Класифікація документів
        "classify_bw_std_thresh":   cfg.getfloat("processing", "classify_bw_std_thresh",   fallback=20.0),
        "classify_edge_ratio_min":  cfg.getfloat("processing", "classify_edge_ratio_min",  fallback=0.03),
        "classify_line_count_min":  cfg.getint("processing",   "classify_line_count_min",  fallback=3),
        # Процентилі авто-яскравості/контрасту
        "auto_percentile_low":  cfg.getfloat("processing", "auto_percentile_low",  fallback=5.0),
        "auto_percentile_high": cfg.getfloat("processing", "auto_percentile_high", fallback=95.0),
        # Бінаризація чб документів
        "bw_binary":         cfg.getboolean("processing", "bw_binary",           fallback=False),
        "jpg_quality":       cfg.getint("output",        "jpg_quality",         fallback=95),
        "save_folder":       cfg.get("output",           "save_folder",         fallback=""),
        "printer_name":      cfg.get("printer",          "printer_name",        fallback="priPrinter"),
    }


def save(settings: dict, path=None):
    """Зберігає словник налаштувань у файл .ini."""
    cfg = configparser.ConfigParser()
    cfg["general"] = {
        "default_mode": settings.get("default_mode", "auto"),
    }
    cfg["processing"] = {
        "autofix_enabled":  str(settings.get("autofix_enabled",  True)).lower(),
        "hdr_in_autofix":   str(settings.get("hdr_in_autofix",   True)).lower(),
        "auto_perspective": str(settings.get("auto_perspective", True)).lower(),
        "sharpen_strength": str(settings.get("sharpen_strength", 0.4)),
        "hdr_strength":     str(settings.get("hdr_strength",     0.5)),
        # Авто-різкість
        "autosharp_threshold":    str(settings.get("autosharp_threshold",    80.0)),
        "autosharp_max_strength": str(settings.get("autosharp_max_strength", 0.7)),
        # Класифікація документів
        "classify_bw_std_thresh":   str(settings.get("classify_bw_std_thresh",   20.0)),
        "classify_edge_ratio_min":  str(settings.get("classify_edge_ratio_min",  0.03)),
        "classify_line_count_min":  str(settings.get("classify_line_count_min",  3)),
        # Процентилі
        "auto_percentile_low":  str(settings.get("auto_percentile_low",  5.0)),
        "auto_percentile_high": str(settings.get("auto_percentile_high", 95.0)),
        # Бінаризація чб
        "bw_binary": str(settings.get("bw_binary", False)).lower(),
    }
    cfg["output"] = {
        "jpg_quality":  str(settings.get("jpg_quality", 95)),
        "save_folder":  settings.get("save_folder", ""),
    }
    cfg["printer"] = {
        "printer_name": settings.get("printer_name", "priPrinter"),
    }

    target = _get_path(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        cfg.write(f)
