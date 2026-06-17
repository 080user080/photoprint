import cv2
import numpy as np
import pytest
from processing import shadow_remove


def _make_doc_with_large_shadow(h=783, w=636, shadow_l=70, paper_l=235):
    """Білий 'документ' з великою темною плямою (тінь) на половину кадру,
    що імітує тінь від руки/телефону. М'який край (Gaussian blur),
    щоб тінь була градієнтною, а не різкою маскою."""
    img_l = np.full((h, w), paper_l, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 4, int(h * 0.75)), (w // 2, h // 3), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    img_l = img_l * (1 - mask) + shadow_l * mask
    img_l = np.clip(img_l, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
    return bgr


def test_large_shadow_no_hard_black_edge():
    """Регресійний тест на баг: тінь, що значно перевищує kernel_size,
    не повинна перетворюватись на різкий чорний контур.

    Перевіряємо: після remove_shadow std яскравості всередині колишньої
    зони тіні близький до std "чистого" паперу (немає залишкового
    різкого перепаду), і немає пікселів, що лишились майже чорними
    (min L не впав нижче розумного порогу)."""
    img = _make_doc_with_large_shadow()
    result = shadow_remove.remove_shadow(img)

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    shadow_zone = l[int(783 * 0.45):, : 636 // 2]
    clean_zone = l[: int(783 * 0.3), :]

    # Старий баг: std у зоні тіні був помітно вищий за std чистого паперу
    # через різкий контур. Допускаємо невеликий запас, але не "вибух".
    assert shadow_zone.std() < clean_zone.std() + 15

    # Не повинно лишатись майже чорних залишкових плям
    assert shadow_zone.min() > 50


def test_small_local_shadow_still_removed():
    """Контроль: дрібна локальна тінь (значно менша за kernel_size)
    і раніше, і зараз має прибиратись майже повністю — другий прохід
    не повинен робити це гірше."""
    h, w = 783, 636
    img_l = np.full((h, w), 235, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (100, 100), 40, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    img_l = img_l * (1 - mask) + 150 * mask
    img_l = np.clip(img_l, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)

    result = shadow_remove.remove_shadow(img)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    shadow_zone = l[60:140, 60:140]
    assert shadow_zone.mean() > 200  # тінь практично прибрана


def test_coarse_pass_can_be_disabled():
    """coarse_pass=False відтворює стару (тільки морфологічну) поведінку —
    параметр не повинен ламати зворотну сумісність для тих, хто хоче
    старий шлях."""
    img = _make_doc_with_large_shadow()
    result_with = shadow_remove.remove_shadow(img, coarse_pass=True)
    result_without = shadow_remove.remove_shadow(img, coarse_pass=False)
    assert not np.array_equal(result_with, result_without)