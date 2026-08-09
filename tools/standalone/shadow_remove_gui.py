"""
shadow_remove_gui.py
Standalone GUI for shadow removal from document images.
Dependencies: opencv-python, numpy, Pillow (pip install opencv-python numpy Pillow)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    import sys
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python numpy Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Shadow Removal Algorithm
# ─────────────────────────────────────────────

COARSE_TARGET_SIDE = 48
PHOTO_COLOR_STD_THRESHOLD = 52   # raised from 30 — less paranoid about colour docs
HSV_SATURATION_RATIO = 0.20
LAPLACIAN_BLUR_THRESHOLD = 10
RANGE_L_THRESHOLD = 210


def _auto_kernel_size(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    short = min(h, w)
    k = int(short * 0.15)
    if k % 2 == 0:
        k += 1
    return max(51, min(k, 301))


def _create_background_model(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return background


def _create_coarse_background(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    scale = COARSE_TARGET_SIDE / min(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return coarse


def _local_std_map(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f ** 2, (ksize, ksize))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(variance)


def _detect_shadow(image: np.ndarray) -> bool:
    """Returns True if shadow is detected and removal should proceed."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    std_a = float(np.std(a_ch))
    std_b = float(np.std(b_ch))
    if std_a > PHOTO_COLOR_STD_THRESHOLD or std_b > PHOTO_COLOR_STD_THRESHOLD:
        return False  # likely a colour photo

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated_ratio = float(np.mean(hsv[:, :, 1] > 50))
    if saturated_ratio > HSV_SATURATION_RATIO:
        return False  # high-saturation content (passport guilloché, etc.)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < LAPLACIAN_BLUR_THRESHOLD:
        return False  # too blurry to process

    range_l = int(l_ch.max()) - int(l_ch.min())
    if range_l > RANGE_L_THRESHOLD:
        return False  # extreme dynamic range — likely not a simple shadow

    std_map = _local_std_map(gray)
    gradient_score = float(np.mean(std_map))
    return gradient_score > 3.0


def remove_shadow(image: np.ndarray,
                  kernel_size: int = 0,
                  coarse_pass: bool = True) -> np.ndarray:
    """
    Remove gradient shadows from a document image.

    Parameters
    ----------
    image       : BGR numpy array
    kernel_size : morphological kernel size (0 = auto)
    coarse_pass : whether to apply the downsample second pass

    Returns
    -------
    Corrected BGR numpy array
    """
    if kernel_size == 0:
        kernel_size = _auto_kernel_size(image)

    channels = cv2.split(image)
    result_channels = []

    for ch in channels:
        # Pass 1 – morphological closing background
        bg1 = _create_background_model(ch, kernel_size)
        normed = cv2.divide(ch, bg1, scale=255)

        if coarse_pass:
            # Pass 2 – coarse downsample background
            bg2 = _create_coarse_background(normed)
            bg2 = np.clip(bg2, 1, 255).astype(np.uint8)
            normed = cv2.divide(normed, bg2, scale=255)

        result_channels.append(normed)

    return cv2.merge(result_channels)


def auto_remove_shadow(image: np.ndarray,
                       force: bool = False) -> tuple[np.ndarray, bool]:
    """
    Automatically detect and remove shadows.

    Returns (processed_image, shadow_was_detected)
    If force=True, skips detection and always processes.
    """
    if force or _detect_shadow(image):
        return remove_shadow(image), True
    return image.copy(), False


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

PREVIEW_MAX = 520   # max pixels for preview panel


def _fit_image_for_preview(img_bgr: np.ndarray) -> ImageTk.PhotoImage:
    h, w = img_bgr.shape[:2]
    scale = min(PREVIEW_MAX / w, PREVIEW_MAX / h, 1.0)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class ShadowRemoverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Shadow Remover")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")

        self._src_path: str | None = None
        self._src_img: np.ndarray | None = None
        self._result_img: np.ndarray | None = None

        # Keep photo references alive
        self._tk_src: ImageTk.PhotoImage | None = None
        self._tk_res: ImageTk.PhotoImage | None = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        PAD = 12
        BG = "#1e1e2e"
        PANEL = "#2a2a3e"
        ACC = "#7c6af7"
        FG = "#cdd6f4"
        ENTRY_BG = "#313244"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", background=PANEL, foreground=FG,
                        font=("Segoe UI", 9, "bold"))
        style.configure("Status.TLabel", background=BG, foreground="#a6adc8",
                        font=("Segoe UI", 9))
        style.configure("Accent.TButton", background=ACC, foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.map("Accent.TButton",
                  background=[("active", "#6a58e0"), ("disabled", "#44475a")])
        style.configure("Ghost.TButton", background=PANEL, foreground=FG,
                        font=("Segoe UI", 9), padding=(8, 5),
                        relief="flat", borderwidth=0)
        style.map("Ghost.TButton",
                  background=[("active", "#3a3a5c")])
        style.configure("TCheckbutton", background=BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.configure("TScale", background=BG, troughcolor=ENTRY_BG,
                        sliderlength=18)
        style.configure("TProgressbar", troughcolor=ENTRY_BG,
                        background=ACC, thickness=6)

        # ── Top bar ──
        top = ttk.Frame(self, style="TFrame", padding=(PAD, PAD, PAD, 4))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Shadow Remover", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="document shadow correction",
                  style="Status.TLabel").pack(side="left", padx=8)

        # ── Main area ──
        main = ttk.Frame(self, style="TFrame", padding=(PAD, 0, PAD, PAD))
        main.grid(row=1, column=0)

        # Left column – controls
        left = ttk.Frame(main, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="ns", padx=(0, PAD))

        # Input section
        ttk.Label(left, text="ВХІДНИЙ ФАЙЛ", style="Sub.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._src_var = tk.StringVar(value="не обрано")
        src_entry = tk.Entry(left, textvariable=self._src_var,
                             width=32, state="readonly",
                             bg=ENTRY_BG, fg=FG, relief="flat",
                             readonlybackground=ENTRY_BG, bd=0,
                             insertbackground=FG, font=("Segoe UI", 9))
        src_entry.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        ttk.Button(left, text="Огляд…", style="Ghost.TButton",
                   command=self._pick_input).grid(row=1, column=1, padx=(4, 0))

        # Output section
        ttk.Label(left, text="ВИХІДНИЙ ФАЙЛ", style="Sub.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(12, 4))

        self._dst_var = tk.StringVar(value="не обрано")
        dst_entry = tk.Entry(left, textvariable=self._dst_var,
                             width=32, state="readonly",
                             bg=ENTRY_BG, fg=FG, relief="flat",
                             readonlybackground=ENTRY_BG, bd=0,
                             insertbackground=FG, font=("Segoe UI", 9))
        dst_entry.grid(row=3, column=0, sticky="ew", pady=(0, 2))
        ttk.Button(left, text="Огляд…", style="Ghost.TButton",
                   command=self._pick_output).grid(row=3, column=1, padx=(4, 0))

        # Separator
        ttk.Separator(left, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=14)

        # Options
        ttk.Label(left, text="НАЛАШТУВАННЯ", style="Sub.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Mode
        ttk.Label(left, text="Режим:", background=PANEL, foreground=FG,
                  font=("Segoe UI", 9)).grid(row=6, column=0, sticky="w")
        self._mode_var = tk.StringVar(value="auto")
        mode_frame = ttk.Frame(left, style="Panel.TFrame")
        mode_frame.grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 10))
        for txt, val in [("Авто", "auto"), ("Завжди", "always"), ("Вимкнено", "off")]:
            tk.Radiobutton(mode_frame, text=txt, variable=self._mode_var,
                           value=val, bg=PANEL, fg=FG, selectcolor="#44475a",
                           activebackground=PANEL, activeforeground=FG,
                           font=("Segoe UI", 9), bd=0,
                           command=self._on_options_change).pack(side="left", padx=(0, 8))

        # Kernel
        ttk.Label(left, text="Ядро морфології (0 = авто):",
                  background=PANEL, foreground=FG,
                  font=("Segoe UI", 9)).grid(row=8, column=0, columnspan=2, sticky="w")
        kernel_row = ttk.Frame(left, style="Panel.TFrame")
        kernel_row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        self._kernel_var = tk.IntVar(value=0)
        self._kernel_label = ttk.Label(kernel_row, text="0 (авто)",
                                       background=PANEL, foreground=ACC,
                                       font=("Segoe UI", 9, "bold"))
        self._kernel_label.pack(side="right")
        ttk.Scale(kernel_row, from_=0, to=301, variable=self._kernel_var,
                  orient="horizontal", length=160,
                  command=self._on_kernel_change).pack(side="left")

        # Coarse pass
        self._coarse_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Другий прохід (coarse downsample)",
                       variable=self._coarse_var,
                       bg=PANEL, fg=FG, selectcolor="#44475a",
                       activebackground=PANEL, activeforeground=FG,
                       font=("Segoe UI", 9), bd=0,
                       command=self._on_options_change).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(0, 4))

        # Progress bar
        self._progress = ttk.Progressbar(left, mode="indeterminate", length=270)
        self._progress.grid(row=11, column=0, columnspan=2,
                            sticky="ew", pady=(14, 6))

        # Status
        self._status_var = tk.StringVar(value="Оберіть вхідний файл")
        ttk.Label(left, textvariable=self._status_var,
                  style="Status.TLabel", wraplength=270).grid(
            row=12, column=0, columnspan=2, sticky="w")

        # Action buttons
        btn_row = ttk.Frame(left, style="Panel.TFrame")
        btn_row.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        self._run_btn = ttk.Button(btn_row, text="▶  Обробити",
                                   style="Accent.TButton",
                                   command=self._run_threaded,
                                   state="disabled")
        self._run_btn.pack(side="left")

        self._save_btn = ttk.Button(btn_row, text="💾  Зберегти",
                                    style="Ghost.TButton",
                                    command=self._save_result,
                                    state="disabled")
        self._save_btn.pack(side="left", padx=(8, 0))

        # Right column – previews
        right = ttk.Frame(main, style="TFrame")
        right.grid(row=0, column=1, sticky="n")

        preview_top = ttk.Frame(right, style="TFrame")
        preview_top.pack()

        # Before
        before_box = ttk.Frame(preview_top, style="Panel.TFrame", padding=8)
        before_box.grid(row=0, column=0, padx=(0, 6))
        ttk.Label(before_box, text="ДО", style="Sub.TLabel").pack()
        self._before_canvas = tk.Canvas(before_box, width=PREVIEW_MAX,
                                        height=PREVIEW_MAX,
                                        bg="#181825", highlightthickness=0)
        self._before_canvas.pack(pady=(4, 0))

        # After
        after_box = ttk.Frame(preview_top, style="Panel.TFrame", padding=8)
        after_box.grid(row=0, column=1)
        ttk.Label(after_box, text="ПІСЛЯ", style="Sub.TLabel").pack()
        self._after_canvas = tk.Canvas(after_box, width=PREVIEW_MAX,
                                       height=PREVIEW_MAX,
                                       bg="#181825", highlightthickness=0)
        self._after_canvas.pack(pady=(4, 0))

        # Size info
        self._size_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._size_var,
                  style="Status.TLabel").pack(pady=(6, 0))

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_kernel_change(self, _=None):
        v = self._kernel_var.get()
        # snap to odd
        if v > 0 and v % 2 == 0:
            v += 1
            self._kernel_var.set(v)
        label = f"{v} (авто)" if v == 0 else str(v)
        self._kernel_label.config(text=label)

    def _on_options_change(self):
        pass  # reserved for live preview in future

    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="Вхідний файл",
            filetypes=[("Зображення", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                       ("Всі файли", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Помилка", f"Не вдалося відкрити:\n{path}")
            return
        self._src_path = path
        self._src_img = img
        self._result_img = None

        # Suggest output path
        base, ext = os.path.splitext(path)
        self._dst_var.set(f"{base}_no_shadow{ext}")

        short = os.path.basename(path)
        self._src_var.set(short if len(short) <= 36 else "…" + short[-33:])

        h, w = img.shape[:2]
        self._size_var.set(f"{w} × {h} px")
        self._show_preview_before(img)
        self._clear_after_canvas()

        self._run_btn.config(state="normal")
        self._save_btn.config(state="disabled")
        self._status_var.set("Готово до обробки")

    def _pick_output(self):
        init = self._dst_var.get()
        if init in ("не обрано", ""):
            init = None
        path = filedialog.asksaveasfilename(
            title="Зберегти як",
            initialfile=os.path.basename(init) if init else "output.jpg",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg"),
                       ("PNG", "*.png"),
                       ("BMP", "*.bmp"),
                       ("TIFF", "*.tiff *.tif"),
                       ("Всі файли", "*.*")])
        if path:
            self._dst_var.set(path)

    def _show_preview_before(self, img: np.ndarray):
        self._tk_src = _fit_image_for_preview(img)
        self._before_canvas.delete("all")
        self._place_image(self._before_canvas, self._tk_src)

    def _show_preview_after(self, img: np.ndarray):
        self._tk_res = _fit_image_for_preview(img)
        self._after_canvas.delete("all")
        self._place_image(self._after_canvas, self._tk_res)

    def _clear_after_canvas(self):
        self._after_canvas.delete("all")
        self._tk_res = None

    def _place_image(self, canvas: tk.Canvas, photo: ImageTk.PhotoImage):
        cw, ch = PREVIEW_MAX, PREVIEW_MAX
        iw, ih = photo.width(), photo.height()
        canvas.create_image((cw - iw) // 2, (ch - ih) // 2,
                            anchor="nw", image=photo)

    # ── Processing ───────────────────────────────────────────────────

    def _run_threaded(self):
        self._run_btn.config(state="disabled")
        self._save_btn.config(state="disabled")
        self._progress.start(12)
        self._status_var.set("Обробка…")
        thread = threading.Thread(target=self._process, daemon=True)
        thread.start()

    def _process(self):
        try:
            img = self._src_img
            mode = self._mode_var.get()
            kernel = self._kernel_var.get()
            if kernel > 0 and kernel % 2 == 0:
                kernel += 1
            coarse = self._coarse_var.get()

            if mode == "off":
                result, detected = img.copy(), False
            elif mode == "always":
                result = remove_shadow(img, kernel_size=kernel, coarse_pass=coarse)
                detected = True
            else:  # auto
                result, detected = auto_remove_shadow(img, force=False)
                if detected and kernel != 0:
                    result = remove_shadow(img, kernel_size=kernel, coarse_pass=coarse)

            self._result_img = result
            status = ("✅ Тіні виявлено та прибрано" if detected
                      else "ℹ️ Тіней не виявлено (авто режим пропустив)")
            if mode == "off":
                status = "⏭ Обробку вимкнено"

            self.after(0, lambda: self._finish(status))
        except Exception as exc:
            self.after(0, lambda: self._error(str(exc)))

    def _finish(self, status: str):
        self._progress.stop()
        self._status_var.set(status)
        self._show_preview_after(self._result_img)
        self._run_btn.config(state="normal")
        self._save_btn.config(state="normal")

    def _error(self, msg: str):
        self._progress.stop()
        self._status_var.set(f"❌ Помилка: {msg}")
        self._run_btn.config(state="normal")
        messagebox.showerror("Помилка обробки", msg)

    # ── Saving ───────────────────────────────────────────────────────

    def _save_result(self):
        if self._result_img is None:
            messagebox.showwarning("Немає результату", "Спочатку запустіть обробку.")
            return

        dst = self._dst_var.get()
        if not dst or dst == "не обрано":
            self._pick_output()
            dst = self._dst_var.get()
        if not dst or dst == "не обрано":
            return

        ext = os.path.splitext(dst)[1].lower()
        params = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        ok = cv2.imwrite(dst, self._result_img, params)
        if ok:
            self._status_var.set(f"✅ Збережено: {os.path.basename(dst)}")
        else:
            messagebox.showerror("Помилка", f"Не вдалося зберегти:\n{dst}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = ShadowRemoverApp()
    app.mainloop()
