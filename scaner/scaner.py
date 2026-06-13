#!/usr/bin/env python3
"""
GUI-обгортка для генерації контексту Python-проєкту (PyQt6).
Вибір папки, гнучке дерево з чекбоксами, налаштування деталізації,
збереження виключень та налаштувань у .ini/.json поруч зі скриптом.
"""

import sys
import os
import json
import ast
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Any
from datetime import datetime

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QModelIndex, QSortFilterProxyModel,
    QStandardPaths, QSettings, QByteArray, QItemSelectionModel
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QTreeView, QPlainTextEdit, QProgressDialog,
    QMessageBox, QLabel, QSplitter, QScrollArea, QMenu, QStyle,
    QHeaderView, QComboBox
)

# ----------------------- Константи -----------------------
SYSTEM_DIRS = {"__pycache__", ".git", ".venv", "venv", ".tox", ".pytest_cache", ".mypy_cache", ".egg-info"}
DOC_EXTENSIONS = {".md", ".txt"}
OTHER_TEXT_EXTENSIONS = {".cfg", ".ini", ".yaml", ".yml", ".toml", ".rst", ".pyi", ".pyw", ".pyx"}
PY_EXTENSIONS = {".py"}

DEFAULT_DETAIL = "medium"  # low, medium, full_doc, high

# ------------------- Функції аналізу Python -------------------
def get_function_signature(node: ast.FunctionDef) -> str:
    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            annotation = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            arg_str += f": {annotation}"
        args.append(arg_str)
    returns = ""
    if node.returns:
        returns = " -> " + (ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns))
    return f"def {node.name}({', '.join(args)}){returns}:"

def get_class_signature(node: ast.ClassDef) -> str:
    bases = []
    for base in node.bases:
        bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
    if bases:
        return f"class {node.name}({', '.join(bases)}):"
    return f"class {node.name}:"

def extract_docstring(node) -> Optional[str]:
    doc = ast.get_docstring(node)
    if doc:
        return doc.split("\n")[0].strip()
    return None

def extract_full_docstring(node) -> Optional[str]:
    return ast.get_docstring(node)

# ------------------- Робочий потік сканування -------------------
class ScannerWorker(QObject):
    finished = pyqtSignal(list, dict)  # список шляхів, словник ієрархії
    progress = pyqtSignal(int)

    def __init__(self, root: Path, show_system: bool, include_docs: bool, include_other_text: bool):
        super().__init__()
        self.root = root
        self.show_system = show_system
        self.include_docs = include_docs
        self.include_other_text = include_other_text

    def run(self):
        file_paths = []
        dir_structure = {}

        for root, dirs, files in os.walk(self.root):
            rel_root = Path(root).relative_to(self.root)
            dirs_to_remove = []
            for d in dirs:
                full_path = Path(root) / d
                if not self.show_system and d in SYSTEM_DIRS:
                    dirs_to_remove.append(d)
            for d in dirs_to_remove:
                dirs.remove(d)

            current_files = []
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in PY_EXTENSIONS:
                    current_files.append(f)
                elif self.include_docs and ext in DOC_EXTENSIONS:
                    current_files.append(f)
                elif self.include_other_text and ext in OTHER_TEXT_EXTENSIONS:
                    current_files.append(f)

            if current_files:
                file_paths.extend([Path(root) / f for f in current_files])

            dir_structure[rel_root.as_posix()] = {
                "dirs": dirs,
                "files": current_files
            }

        dir_structure = {k: v for k, v in dir_structure.items() if v["files"] or v["dirs"]}
        self.finished.emit([str(p) for p in sorted(file_paths)], dir_structure)

# ------------------- Робочий потік генерації звіту -------------------
class GeneratorWorker(QObject):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, selected_files: List[str], root: Path, configs: List[dict]):
        super().__init__()
        self.selected_files = selected_files
        self.root = root
        self.configs = configs

    def run(self):
        results = []
        total_files = len(self.selected_files)
        total_steps = len(self.configs) * total_files
        current_step = 0

        for config in self.configs:
            report_lines = []
            report_lines.append("СТРУКТУРА ПРОЄКТУ")
            report_lines.append("=" * 60)
            report_lines.append(str(self.root))
            
            # Фільтруємо файли для структури згідно з поточним конфігом
            valid_files = []
            for f in self.selected_files:
                ext = Path(f).suffix.lower()
                if ext in DOC_EXTENSIONS and not config.get("docs", False):
                    continue
                if ext in OTHER_TEXT_EXTENSIONS and not config.get("other", False):
                    continue
                valid_files.append(f)
                rel = Path(f).relative_to(self.root)
                report_lines.append(str(rel))

            report_lines.append("\n" + "=" * 60)
            report_lines.append(f"ДЕТАЛІЗАЦІЯ КОДУ (рівень: {config['detail']})")
            report_lines.append("=" * 60)

            for file_path in valid_files:
                current_step += 1
                self.progress.emit(int((current_step / total_steps) * 100))
                report_lines.append(self._process_file(Path(file_path), config))

            results.append({
                "filename": config["filename"],
                "content": "\n".join(report_lines)
            })

        self.progress.emit(100)
        self.finished.emit(results)

    def _process_file(self, file_path: Path, config: dict) -> str:
        ext = file_path.suffix.lower()
        if ext in PY_EXTENSIONS:
            return self._process_py(file_path, config)
        elif ext in DOC_EXTENSIONS or ext in OTHER_TEXT_EXTENSIONS:
            if config["detail"] == "low":
                return f"# Пропущено (low деталізація): {file_path}\n"
            return self._process_text(file_path)
        else:
            return f"# Невідомий тип: {file_path}\n"

    def _process_py(self, file_path: Path, config: dict) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except SyntaxError:
            return f"# [Помилка синтаксису] {file_path}\n"
        except Exception as e:
            return f"# [Помилка читання] {file_path}: {e}\n"

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Файл: {file_path}")
        lines.append(f"{'='*60}")

        if config["detail"] == "high":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            return "\n".join(lines)

        if config.get("imports", False):
            imports = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    imports.append(f"from {module} import {names}")
            if imports:
                lines.append("Імпорти:")
                lines.extend(f"  {imp}" for imp in imports)
                lines.append("")

        entities = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_lines = [get_class_signature(node)]
                if config["detail"] == "medium":
                    doc = extract_docstring(node)
                    if doc:
                        class_lines.append(f'    """{doc}"""')
                elif config["detail"] == "full_doc":
                    doc = extract_full_docstring(node)
                    if doc:
                        class_lines.append(f'    """{doc}"""')
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        sig = get_function_signature(item)
                        method_lines = [f"    {sig}"]
                        if config["detail"] == "medium":
                            doc = extract_docstring(item)
                            if doc:
                                method_lines.append(f'        """{doc}"""')
                        elif config["detail"] == "full_doc":
                            doc = extract_full_docstring(item)
                            if doc:
                                method_lines.append(f'        """{doc}"""')
                        class_lines.append("\n".join(method_lines))
                entities.append("\n".join(class_lines))
            elif isinstance(node, ast.FunctionDef):
                sig = get_function_signature(node)
                func_lines = [sig]
                if config["detail"] == "medium":
                    doc = extract_docstring(node)
                    if doc:
                        func_lines.append(f'    """{doc}"""')
                elif config["detail"] == "full_doc":
                    doc = extract_full_docstring(node)
                    if doc:
                        func_lines.append(f'    """{doc}"""')
                entities.append("\n".join(func_lines))

        if entities:
            lines.append("Сутності:")
            lines.extend(entities)
        else:
            lines.append("(немає класів чи функцій верхнього рівня)")

        return "\n".join(lines)

    def _process_text(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"# [Помилка читання] {file_path}: {e}\n"

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Документація/файл: {file_path}")
        lines.append(f"{'='*60}")
        lines.append(content)
        return "\n".join(lines)

# ------------------- Головне вікно -------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project Context Generator")
        self.setMinimumSize(900, 650)

        self.settings_path = Path(__file__).parent / "settings.ini"
        self.exclusions_path = Path(__file__).parent / "exclusions.json"

        self.settings = QSettings(str(self.settings_path), QSettings.Format.IniFormat)
        self.exclusions_cache: Dict[str, List[str]] = self._load_exclusions()

        self.project_root: Optional[Path] = None
        self.file_tree_model: Optional[QStandardItemModel] = None
        self.current_file_paths: List[str] = []
        self.full_report: Optional[str] = None

        self.init_ui()
        self.restore_settings()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ==== Вибір проєкту ====
        project_group = QGroupBox("Проєкт")
        project_layout = QHBoxLayout()
        self.project_edit = QLineEdit()
        self.project_edit.setReadOnly(True)
        project_layout.addWidget(QLabel("Папка:"))
        project_layout.addWidget(self.project_edit)
        browse_btn = QPushButton("Огляд...")
        browse_btn.clicked.connect(self.browse_project)
        project_layout.addWidget(browse_btn)
        self.scan_btn = QPushButton("Сканувати")
        self.scan_btn.clicked.connect(self.scan_project)
        project_layout.addWidget(self.scan_btn)
        project_group.setLayout(project_layout)
        main_layout.addWidget(project_group)

        # ==== Параметри ====
        params_group = QGroupBox("Параметри")
        params_layout = QHBoxLayout()

        detail_layout = QVBoxLayout()
        detail_layout.addWidget(QLabel("Деталізація:"))
        self.detail_group = QButtonGroup(self)
        self.radio_low = QRadioButton("Low")
        self.radio_medium = QRadioButton("Medium")
        self.radio_full_doc = QRadioButton("Full Doc")
        self.radio_high = QRadioButton("High")
        self.detail_group.addButton(self.radio_low, 0)
        self.detail_group.addButton(self.radio_medium, 1)
        self.detail_group.addButton(self.radio_full_doc, 2)
        self.detail_group.addButton(self.radio_high, 3)
        detail_layout.addWidget(self.radio_low)
        detail_layout.addWidget(self.radio_medium)
        detail_layout.addWidget(self.radio_full_doc)
        detail_layout.addWidget(self.radio_high)
        self.radio_medium.setChecked(True)
        
        # Чекбокс генерації всіх варіантів
        self.generate_all_check = QCheckBox("Створити всі варіанти звітів (Batch)")
        self.generate_all_check.setToolTip("Генерує 12 файлів (усі рівні + варіанти з імпортами/документацією)")
        detail_layout.addWidget(self.generate_all_check)
        params_layout.addLayout(detail_layout)

        options_layout = QVBoxLayout()
        self.imports_check = QCheckBox("Включити імпорти")
        self.system_dirs_check = QCheckBox("Показувати системні папки")
        self.docs_check = QCheckBox("Документація (.md, .txt)")
        self.docs_check.setChecked(True)
        self.other_text_check = QCheckBox("Інші текстові файли")
        options_layout.addWidget(self.imports_check)
        options_layout.addWidget(self.system_dirs_check)
        options_layout.addWidget(self.docs_check)
        options_layout.addWidget(self.other_text_check)
        params_layout.addLayout(options_layout)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # ==== Дерево файлів ====
        self.tree_group = QGroupBox("Файли для індексації")
        tree_layout = QVBoxLayout()
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Пошук:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Введіть частину назви...")
        self.search_edit.textChanged.connect(self.filter_tree)
        search_layout.addWidget(self.search_edit)
        tree_layout.addLayout(search_layout)

        # Панель керування деревом
        tree_controls_layout = QHBoxLayout()
        
        self.expand_check = QCheckBox("Розгорнути все")
        self.expand_check.setChecked(True)
        self.expand_check.toggled.connect(self.on_expand_toggled)
        tree_controls_layout.addWidget(self.expand_check)

        self.selection_combo = QComboBox()
        self.selection_combo.addItems(["Відновити попередній вибір", "Відмітити все", "Зняти всі"])
        self.selection_combo.currentIndexChanged.connect(self.on_selection_combo_changed)
        tree_controls_layout.addWidget(self.selection_combo)
        
        tree_controls_layout.addStretch()
        tree_layout.addLayout(tree_controls_layout)

        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.on_tree_context_menu)
        tree_layout.addWidget(self.tree_view)

        self.tree_group.setLayout(tree_layout)
        self.tree_group.setVisible(False)
        main_layout.addWidget(self.tree_group)

        # ==== Результат ====
        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Вихідний файл:"))
        self.output_edit = QLineEdit()
        output_layout.addWidget(self.output_edit)
        save_btn = QPushButton("Зберегти як...")
        save_btn.clicked.connect(self.choose_output_file)
        output_layout.addWidget(save_btn)
        result_layout.addLayout(output_layout)

        save_opts_layout = QHBoxLayout()
        self.remember_output_check = QCheckBox("Запам'ятати шлях")
        self.auto_save_check = QCheckBox("Автозбереження в папку проєкту")
        save_opts_layout.addWidget(self.remember_output_check)
        save_opts_layout.addWidget(self.auto_save_check)
        result_layout.addLayout(save_opts_layout)

        action_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Згенерувати")
        self.generate_btn.clicked.connect(self.generate_report)
        action_layout.addWidget(self.generate_btn)
        self.copy_btn = QPushButton("Копіювати в буфер обміну")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        action_layout.addWidget(self.copy_btn)
        result_layout.addLayout(action_layout)

        result_layout.addWidget(QLabel("Прев'ю (перші 100 рядків):"))
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumBlockCount(100)
        result_layout.addWidget(self.preview_text)

        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

    # ------------------- Слоти -------------------
    def browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Виберіть папку проєкту")
        if path:
            self.project_edit.setText(path)

    def scan_project(self):
        path = self.project_edit.text()
        if not path:
            QMessageBox.warning(self, "Помилка", "Виберіть папку проєкту.")
            return
        self.project_root = Path(path).resolve()
        self.scan_btn.setEnabled(False)
        progress = QProgressDialog("Сканування проєкту...", "Скасувати", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.scanner_worker = ScannerWorker(
            self.project_root,
            self.system_dirs_check.isChecked(),
            self.docs_check.isChecked(),
            self.other_text_check.isChecked()
        )
        self.scanner_thread = QThread()
        self.scanner_worker.moveToThread(self.scanner_thread)
        self.scanner_thread.started.connect(self.scanner_worker.run)
        self.scanner_worker.finished.connect(self.on_scan_finished)
        self.scanner_worker.finished.connect(self.scanner_thread.quit)
        self.scanner_worker.finished.connect(progress.close)
        self.scanner_thread.start()

    def on_scan_finished(self, file_paths: List[str], dir_structure: dict):
        self.current_file_paths = file_paths
        
        # Скидаємо комбобокс виділення без тригерування сигналу
        self.selection_combo.blockSignals(True)
        self.selection_combo.setCurrentIndex(0)
        self.selection_combo.blockSignals(False)
        
        self.build_tree(dir_structure)
        self.tree_group.setVisible(True)
        self.scan_btn.setEnabled(True)

    def build_tree(self, dir_structure: dict):
        model = QStandardItemModel()
        root_item = model.invisibleRootItem()

        project_hash = hashlib.sha1(str(self.project_root).encode()).hexdigest()
        excluded_relative = set(self.exclusions_cache.get(project_hash, []))

        def add_item(parent: QStandardItem, name: str, is_dir: bool, rel_path: str) -> QStandardItem:
            item = QStandardItem(name)
            item.setCheckable(True)
            checked = rel_path not in excluded_relative
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setEditable(False)
            if is_dir:
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            else:
                ext = Path(name).suffix.lower()
                if ext in PY_EXTENSIONS:
                    icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
                elif ext in DOC_EXTENSIONS:
                    icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
                else:
                    icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            item.setIcon(icon)
            item.setData(rel_path, Qt.ItemDataRole.UserRole)
            parent.appendRow(item)
            return item

        def fill_recursive(parent_item: QStandardItem, rel_dir: str):
            info = dir_structure.get(rel_dir, None)
            if info is None:
                return
            for d in sorted(info["dirs"]):
                dir_rel = (Path(rel_dir) / d).as_posix()
                dir_item = add_item(parent_item, d, True, dir_rel)
                fill_recursive(dir_item, dir_rel)
            for f in sorted(info["files"]):
                file_rel = (Path(rel_dir) / f).as_posix()
                add_item(parent_item, f, False, file_rel)

        root_info = dir_structure.get(".", {"dirs": [], "files": []})
        for d in sorted(root_info.get("dirs", [])):
            dir_rel = d
            dir_item = add_item(root_item, d, True, dir_rel)
            fill_recursive(dir_item, dir_rel)
        for f in sorted(root_info.get("files", [])):
            add_item(root_item, f, False, f)

        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(model)
        self.proxy_model.setRecursiveFilteringEnabled(True)
        self.tree_view.setModel(self.proxy_model)

        if self.expand_check.isChecked():
            self.tree_view.expandAll()

        self.file_tree_model = model
        model.itemChanged.connect(self.on_item_changed)

    def on_expand_toggled(self, checked: bool):
        if checked:
            self.tree_view.expandAll()
        else:
            self.tree_view.collapseAll()

    def on_selection_combo_changed(self, index: int):
        if not self.file_tree_model:
            return
        if index == 1:
            self._set_all_checkstate(Qt.CheckState.Checked)
        elif index == 2:
            self._set_all_checkstate(Qt.CheckState.Unchecked)
        elif index == 0:
            self.apply_saved_selection()

    def apply_saved_selection(self):
        if not self.file_tree_model or not self.project_root:
            return
        project_hash = hashlib.sha1(str(self.project_root).encode()).hexdigest()
        excluded_relative = set(self.exclusions_cache.get(project_hash, []))

        self.file_tree_model.itemChanged.disconnect(self.on_item_changed)
        root = self.file_tree_model.invisibleRootItem()

        def apply_to_node(item: QStandardItem):
            for row in range(item.rowCount()):
                child = item.child(row)
                rel = child.data(Qt.ItemDataRole.UserRole)
                if rel and child.isCheckable():
                    if rel in excluded_relative:
                        child.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        child.setCheckState(Qt.CheckState.Checked)
                apply_to_node(child)

        apply_to_node(root)
        self._update_all_parents(root)
        self.file_tree_model.itemChanged.connect(self.on_item_changed)

    def on_item_changed(self, item: QStandardItem):
        if not item.isCheckable():
            return
        self.file_tree_model.itemChanged.disconnect(self.on_item_changed)
        state = item.checkState()
        self._set_children_checkstate(item, state)
        parent = item.parent()
        if parent and parent.isCheckable():
            self._update_parent_checkstate(parent)
        self.file_tree_model.itemChanged.connect(self.on_item_changed)

    def _set_children_checkstate(self, item: QStandardItem, state: Qt.CheckState):
        for row in range(item.rowCount()):
            child = item.child(row)
            if child.isCheckable():
                child.setCheckState(state)
                self._set_children_checkstate(child, state)

    def _update_parent_checkstate(self, parent: QStandardItem):
        checked = 0
        total = 0
        for row in range(parent.rowCount()):
            child = parent.child(row)
            if child.isCheckable():
                total += 1
                if child.checkState() == Qt.CheckState.Checked:
                    checked += 1
                elif child.checkState() == Qt.CheckState.PartiallyChecked:
                    checked += 0.5
        if total == 0:
            new_state = Qt.CheckState.Unchecked
        elif checked == total:
            new_state = Qt.CheckState.Checked
        elif checked > 0:
            new_state = Qt.CheckState.PartiallyChecked
        else:
            new_state = Qt.CheckState.Unchecked
        parent.setCheckState(new_state)
        grandparent = parent.parent()
        if grandparent and grandparent.isCheckable():
            self._update_parent_checkstate(grandparent)

    def filter_tree(self, text: str):
        if hasattr(self, 'proxy_model'):
            self.proxy_model.setFilterFixedString(text)

    def on_tree_context_menu(self, pos):
        menu = QMenu()
        select_all = menu.addAction("Вибрати всі")
        deselect_all = menu.addAction("Зняти всі")
        invert = menu.addAction("Інвертувати виділення")
        action = menu.exec(self.tree_view.viewport().mapToGlobal(pos))
        if action == select_all:
            self._set_all_checkstate(Qt.CheckState.Checked)
            self.selection_combo.setCurrentIndex(1)
        elif action == deselect_all:
            self._set_all_checkstate(Qt.CheckState.Unchecked)
            self.selection_combo.setCurrentIndex(2)
        elif action == invert:
            self._invert_checkstates()

    def _set_all_checkstate(self, state: Qt.CheckState):
        if not self.file_tree_model:
            return
        self.file_tree_model.itemChanged.disconnect(self.on_item_changed)
        root = self.file_tree_model.invisibleRootItem()
        self._set_children_checkstate(root, state)
        self.file_tree_model.itemChanged.connect(self.on_item_changed)

    def _invert_checkstates(self):
        if not self.file_tree_model:
            return
        self.file_tree_model.itemChanged.disconnect(self.on_item_changed)
        root = self.file_tree_model.invisibleRootItem()
        self._invert_children(root)
        self._update_all_parents(root)
        self.file_tree_model.itemChanged.connect(self.on_item_changed)

    def _invert_children(self, item: QStandardItem):
        for row in range(item.rowCount()):
            child = item.child(row)
            if child.isCheckable():
                new_state = Qt.CheckState.Unchecked if child.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                child.setCheckState(new_state)
            self._invert_children(child)

    def _update_all_parents(self, item: QStandardItem):
        for row in range(item.rowCount()):
            child = item.child(row)
            self._update_all_parents(child)
        if item.isCheckable():
            self._update_parent_checkstate(item)

    def _get_current_detail(self) -> str:
        if self.radio_low.isChecked():
            return "low"
        elif self.radio_medium.isChecked():
            return "medium"
        elif self.radio_full_doc.isChecked():
            return "full_doc"
        elif self.radio_high.isChecked():
            return "high"
        return "medium"

    def _generate_filename(self) -> str:
        detail = self._get_current_detail().capitalize()
        parts = [detail]
        if self.imports_check.isChecked():
            parts.append("imp")
        if self.docs_check.isChecked():
            parts.append("doc")
        if self.other_text_check.isChecked():
            parts.append("other")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(f"context_{timestamp}")
        return "_".join(parts) + ".txt"

    def choose_output_file(self):
        default_name = self._generate_filename()
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти звіт", default_name,
                                               "Text Files (*.txt);;All Files (*)")
        if path:
            self.output_edit.setText(path)

    def generate_report(self):
        if not self.project_root or not self.file_tree_model:
            QMessageBox.warning(self, "Помилка", "Спочатку виконайте сканування проєкту.")
            return

        selected_files = self._get_checked_files()
        if not selected_files:
            QMessageBox.warning(self, "Помилка", "Не вибрано жодного файлу.")
            return

        configs = []
        is_batch = self.generate_all_check.isChecked()

        if is_batch:
            levels = ["low", "medium", "full_doc", "high"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for lvl in levels:
                configs.append({
                    "detail": lvl, "imports": False, "docs": False, "other": False,
                    "filename": f"{lvl.capitalize()}_base_context.txt"
                })
                configs.append({
                    "detail": lvl, "imports": True, "docs": False, "other": False,
                    "filename": f"{lvl.capitalize()}_imports_context.txt"
                })
                configs.append({
                    "detail": lvl, "imports": True, "docs": True, "other": True,
                    "filename": f"{lvl.capitalize()}_full_mix_context.txt"
                })
        else:
            configs.append({
                "detail": self._get_current_detail(),
                "imports": self.imports_check.isChecked(),
                "docs": self.docs_check.isChecked(),
                "other": self.other_text_check.isChecked(),
                "filename": self._generate_filename()
            })

        output_path = self.output_edit.text() if not is_batch else None
        if not is_batch and self.auto_save_check.isChecked() and self.project_root:
            output_path = str(self.project_root / configs[0]["filename"])
            self.output_edit.setText(output_path)

        self.generate_btn.setEnabled(False)
        progress = QProgressDialog("Генерація звітів...", "Скасувати", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.generator_worker = GeneratorWorker(selected_files, self.project_root, configs)
        self.generator_thread = QThread()
        self.generator_worker.moveToThread(self.generator_thread)
        self.generator_thread.started.connect(self.generator_worker.run)
        self.generator_worker.progress.connect(progress.setValue)
        
        self.generator_worker.finished.connect(lambda res: self.on_generation_finished(res, output_path, is_batch))
        self.generator_worker.finished.connect(self.generator_thread.quit)
        self.generator_worker.finished.connect(progress.close)
        self.generator_thread.start()

    def on_generation_finished(self, results: List[dict], user_output_path: Optional[str], is_batch: bool):
        if not results:
            self.generate_btn.setEnabled(True)
            return

        if not is_batch:
            self.full_report = results[0]["content"]
            lines = self.full_report.splitlines()
            self.preview_text.setPlainText("\n".join(lines[:100]))
            
            if user_output_path:
                try:
                    with open(user_output_path, "w", encoding="utf-8") as f:
                        f.write(self.full_report)
                    QMessageBox.information(self, "Успіх", f"Звіт збережено у {user_output_path}")
                except Exception as e:
                    QMessageBox.warning(self, "Помилка", f"Не вдалося зберегти файл: {e}")
            else:
                QMessageBox.information(self, "Готово", "Звіт згенеровано. Ви можете зберегти його вручну або скопіювати.")
        else:
            self.full_report = results[-1]["content"] 
            self.preview_text.setPlainText("Згенеровано пакет звітів. Відображено останній:\n\n" + "\n".join(self.full_report.splitlines()[:100]))
            
            save_dir = self.project_root if self.auto_save_check.isChecked() else Path(QFileDialog.getExistingDirectory(self, "Виберіть папку для збереження всіх звітів"))
            
            if save_dir:
                save_dir = Path(save_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = save_dir / f"ContextReports_{timestamp}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    for res in results:
                        file_path = batch_dir / res["filename"]
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(res["content"])
                    QMessageBox.information(self, "Успіх", f"Всі {len(results)} звітів збережено у папку:\n{batch_dir}")
                except Exception as e:
                    QMessageBox.warning(self, "Помилка", f"Сталася помилка при збереженні: {e}")

        self.generate_btn.setEnabled(True)
        self._save_exclusions()

    def _get_checked_files(self) -> List[str]:
        checked = []
        if not self.file_tree_model:
            return checked
        root = self.file_tree_model.invisibleRootItem()
        self._collect_checked(root, checked)
        return checked

    def _collect_checked(self, item: QStandardItem, result: List[str]):
        for row in range(item.rowCount()):
            child = item.child(row)
            if child.isCheckable() and child.checkState() != Qt.CheckState.Checked:
                continue
            if child.rowCount() == 0:
                rel_path = child.data(Qt.ItemDataRole.UserRole)
                if rel_path:
                    full = (self.project_root / rel_path).as_posix()
                    result.append(full)
            else:
                self._collect_checked(child, result)

    def copy_to_clipboard(self):
        if self.full_report:
            QApplication.clipboard().setText(self.full_report)
            QMessageBox.information(self, "Скопійовано", "Звіт скопійовано в буфер обміну.")
        else:
            QMessageBox.warning(self, "Помилка", "Спочатку згенеруйте звіт.")

    # ------------------- Збереження/відновлення налаштувань -------------------
    def save_settings(self):
        self.settings.setValue("LastSession/project_dir", self.project_edit.text())
        self.settings.setValue("LastSession/detail_level", self._get_current_detail())
        self.settings.setValue("LastSession/include_imports", self.imports_check.isChecked())
        self.settings.setValue("LastSession/show_system_dirs", self.system_dirs_check.isChecked())
        self.settings.setValue("LastSession/include_docs", self.docs_check.isChecked())
        self.settings.setValue("LastSession/include_other_text", self.other_text_check.isChecked())
        self.settings.setValue("LastSession/generate_all", self.generate_all_check.isChecked())
        self.settings.setValue("LastSession/output_path", self.output_edit.text())
        self.settings.setValue("LastSession/remember_output", self.remember_output_check.isChecked())
        self.settings.setValue("LastSession/auto_save", self.auto_save_check.isChecked())
        self.settings.setValue("Window/geometry", self.saveGeometry())
        self.settings.setValue("Window/state", self.saveState())

    def restore_settings(self):
        if self.settings.contains("LastSession/project_dir"):
            self.project_edit.setText(self.settings.value("LastSession/project_dir", ""))
        detail = self.settings.value("LastSession/detail_level", "medium")
        for btn in self.detail_group.buttons():
            if btn.text().lower().replace(" ", "_") == detail:
                btn.setChecked(True)
                break
        self.imports_check.setChecked(self.settings.value("LastSession/include_imports", False) == "true")
        self.system_dirs_check.setChecked(self.settings.value("LastSession/show_system_dirs", False) == "true")
        self.docs_check.setChecked(self.settings.value("LastSession/include_docs", True) != "false")
        self.other_text_check.setChecked(self.settings.value("LastSession/include_other_text", False) == "true")
        self.generate_all_check.setChecked(self.settings.value("LastSession/generate_all", False) == "true")
        self.output_edit.setText(self.settings.value("LastSession/output_path", ""))
        self.remember_output_check.setChecked(self.settings.value("LastSession/remember_output", False) == "true")
        self.auto_save_check.setChecked(self.settings.value("LastSession/auto_save", False) == "true")
        if self.settings.contains("Window/geometry"):
            self.restoreGeometry(self.settings.value("Window/geometry", QByteArray()))
        if self.settings.contains("Window/state"):
            self.restoreState(self.settings.value("Window/state", QByteArray()))

    def closeEvent(self, event):
        self.save_settings()
        event.accept()

    # ------------------- Робота з виключеннями -------------------
    def _load_exclusions(self) -> Dict[str, List[str]]:
        if self.exclusions_path.exists():
            try:
                with open(self.exclusions_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_exclusions(self):
        if not self.project_root or not self.file_tree_model:
            return
        project_hash = hashlib.sha1(str(self.project_root).encode()).hexdigest()
        excluded = []
        root = self.file_tree_model.invisibleRootItem()
        self._collect_unchecked(root, excluded)
        self.exclusions_cache[project_hash] = excluded
        try:
            with open(self.exclusions_path, "w", encoding="utf-8") as f:
                json.dump(self.exclusions_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Не вдалося зберегти виключення: {e}")

    def _collect_unchecked(self, item: QStandardItem, result: List[str]):
        for row in range(item.rowCount()):
            child = item.child(row)
            if child.isCheckable() and child.checkState() == Qt.CheckState.Unchecked:
                rel = child.data(Qt.ItemDataRole.UserRole)
                if rel:
                    result.append(rel)
            self._collect_unchecked(child, result)

# ------------------- Запуск -------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())