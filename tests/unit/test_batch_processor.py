"""
Тести для BatchProcessor.sync_new_files (TODO 2.5-b).

Проблема: set_files скидає _index, через що файли, додані в GUI-чергу
після часткового проходу, не потрапляли в _processor і "губились"
при ручному друці.

sync_new_files має додавати нові шляхи без скидання позиції друку.
"""

import os

from batch.batch_processor import BatchProcessor


# ---------------------------------------------------------------------------
# Допоміжне
# ---------------------------------------------------------------------------

def _make_processor(paths: list[str]) -> BatchProcessor:
    """Створює BatchProcessor з заданим початковим списком файлів."""
    p = BatchProcessor({})
    p.set_files(paths)
    return p


def _jpg_paths(*names: str) -> list[str]:
    """Формує псевдо-шляхи з .jpg розширенням (файли можуть не існувати)."""
    return [os.path.join("C:/test", f"{n}.jpg") for n in names]


# ---------------------------------------------------------------------------
# Базові перевірки sync_new_files
# ---------------------------------------------------------------------------

def test_adds_new_files_without_resetting_index():
    """Нові файли додаються, _index не скидається."""
    p = _make_processor(_jpg_paths("a", "b", "c"))
    p.skip_current()                      # _index: 0 -> 1
    assert p.current_index == 1

    p.sync_new_files(_jpg_paths("d", "e"))

    assert p.total == 5
    assert p.current_index == 1           # позиція збережена
    assert p.files == _jpg_paths("a", "b", "c", "d", "e")


def test_does_not_add_duplicates():
    """Існуючі шляхи не дублюються."""
    p = _make_processor(_jpg_paths("a", "b"))

    p.sync_new_files(_jpg_paths("a", "b", "c"))

    assert p.total == 3
    assert p.files == _jpg_paths("a", "b", "c")


def test_empty_list_no_change():
    """Порожній список — стан не змінюється."""
    p = _make_processor(_jpg_paths("a", "b"))
    p.skip_current()                      # _index: 0 -> 1

    p.sync_new_files([])

    assert p.total == 2
    assert p.current_index == 1
    assert p.files == _jpg_paths("a", "b")


def test_unsupported_files_filtered():
    """Непідтримувані розширення відфільтровуються."""
    all_files = _jpg_paths("a") + ["C:/test/b.txt", "C:/test/c.pdf"]
    p = _make_processor(_jpg_paths("a"))
    p.skip_current()                      # _index: 0 -> 1

    p.sync_new_files(all_files)

    assert p.total == 1                   # тільки a.jpg лишився
    assert p.current_index == 1
    assert p.files == _jpg_paths("a")


def test_empty_processor_equivalent_to_set_files():
    """Якщо процесор порожній — sync_new_files еквівалентний set_files."""
    p = BatchProcessor({})

    p.sync_new_files(_jpg_paths("a", "b"))

    assert p.total == 2
    assert p.current_index == 0
    assert p.has_next() is True
    assert p.files == _jpg_paths("a", "b")


def test_sync_new_files_after_skip_current_keeps_position():
    """Після skip_current позиція зберігається, нові файли доступні."""
    p = _make_processor(_jpg_paths("a", "b", "c"))
    p.skip_current()                      # a пропущено, _index = 1

    p.sync_new_files(_jpg_paths("d"))

    assert p.current_index == 1
    assert p.current_file() == _jpg_paths("b")[0]   # наступний — b
    assert p.total == 4


# ---------------------------------------------------------------------------
# Сценарій із TODO 2.5-b
# ---------------------------------------------------------------------------

def test_todo_scenario_partial_batch_then_new_files():
    """
    Сценарій з TODO: 5 з 10 пройдено → додано 3 нових файли в чергу →
    _index лишається 5, total стає 13, has_next() — True.
    """
    initial = _jpg_paths(*[f"f{i}" for i in range(10)])
    p = _make_processor(initial)

    # Симулюємо прохід 5 файлів через skip_current (як у ручному режимі).
    # Не використовуємо print_current — він іде на фізичний друк,
    # а тести друку потребують явного підтвердження людини (project_rules).
    for _ in range(5):
        p.skip_current()

    assert p.current_index == 5
    assert p.total == 10
    assert p.has_next() is True

    # Додаємо 3 нових файли (як користувач додав би в GUI-чергу)
    new_files = _jpg_paths("added1", "added2", "added3")
    p.sync_new_files(new_files)

    assert p.total == 13
    assert p.current_index == 5           # позиція не скинута
    assert p.has_next() is True
    # Поточний файл — 6-й з вихідного списку (індекс 5)
    assert p.current_file() == initial[5]
    # Нові файли тепер у кінці черги
    assert p.files[10:] == new_files
    # Прохід до кінця: після ще 5 старих маємо 3 нових
    for _ in range(5):
        p.skip_current()
    assert p.current_index == 10
    assert p.current_file() == new_files[0]
    assert p.has_next() is True


def test_completed_queue_then_new_files():
    """
    Чергу повністю пройдено (_index == total) → додано нові файли →
    вони доступні через has_next() і друк не починається з нуля.
    """
    p = _make_processor(_jpg_paths("a", "b"))
    p.skip_current()
    p.skip_current()
    assert p.current_index == 2
    assert p.has_next() is False          # черга завершена

    p.sync_new_files(_jpg_paths("c", "d"))

    assert p.total == 4
    assert p.current_index == 2           # не скинуто
    assert p.has_next() is True           # нові файли доступні
    assert p.current_file() == _jpg_paths("c")[0]


def test_sync_new_files_does_not_reattach_processed_files():
    """Раніше оброблені файли не друкуються повторно після sync."""
    p = _make_processor(_jpg_paths("a", "b", "c"))
    p.skip_current()                      # a пропущено/надруковано

    p.sync_new_files(_jpg_paths("a", "b", "c", "d"))  # старі + один новий
    assert p.current_index == 1
    assert p.current_file() == _jpg_paths("b")[0]     # b — наступний, а не a
    assert p.total == 4