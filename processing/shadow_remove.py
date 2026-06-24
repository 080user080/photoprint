"""
Видалення тіней з документів через background estimation.
Метод: морфологічне закриття (MORPH_CLOSE) + легке GaussianBlur для оцінки фону,
       потім cv2.divide для нормалізації освітлення.
       + ДРУГИЙ, "грубий" прохід (downsample-based) для великих тіней,
       які один морфологічний kernel принципово не може прибрати (див. нижче).

Чому морфологічний метод замість чистого GaussianBlur:
- GaussianBlur — low-pass фільтр, який «розмазує» яскравість радіально від краю тіні,
  створюючи плавний градієнт шириною ~kernel_size/2. Після cv2.divide це дає
  світлу/темну облямівку («ореол») навколо різких тіней.
- MORPH_CLOSE з еліптичним ядром: dilate заповнює темну тінь значеннями світлого фону,
  потім erode повертає границю — геометрія краю зберігається, ореолів немає.
- Еліптичне ядро обрано замість прямокутного, бо кругова форма не створює
  «сходинок» на кутах тіней.
- Легке GaussianBlur (мале ядро ~kernel_size/5) поверх результату прибирає
  текстуру документа (текст, лінії), щоб вона не потрапила в модель фону.

ВАЖЛИВО — чому одного MORPH_CLOSE недостатньо для ВЕЛИКИХ тіней
(наприклад тінь від руки/телефону, що закриває половину кадру):

  MORPH_CLOSE може або повністю "не дотягнутись", або стати по суті
  no-op'ом — залежно від співвідношення розміру ядра й розміру тіні.
  Це не помилка реалізації, а математична межа методу:

  - kernel_size << ширина тіні:
      dilate всередині тіні не дістає до світлих пікселів фону (вони
      задалеко), тому центр тіні в моделі фону залишається темним.
      Але рівно по краю тіні, на відстані ~kernel_size/2 від реального
      контуру, dilate ВЖЕ дістає світлий фон — модель фону різко
      "перескакує" з темного на світле саме там. Після cv2.divide
      (оригінал лишається темним, фон раптом світлий) це дає НЕ
      градієнт, а різкий темний контур — саме той "чіткий чорний
      контур телефону", про який скаржаться користувачі. Поза цим
      контуром (де kernel дотягнувся скрізь) корекція спрацьовує добре
      і фон стає білим — тож контраст контуру щодо побілілого фону
      стає ще більш разючим.
  - kernel_size >> ширина тіні:
      dilate з такого ядра дістає світлий фон практично з будь-якої
      точки кадру (бо тінь відносно ядра — маленька крапка), тому
      модель фону стає майже однорідною константою по всьому
      зображенню. Ділення на константу — це просто масштабування
      яскравості, воно НЕ прибирає форму тіні. Тінь залишається
      майже незмінною (перевірено експериментально: kernel=401 на
      636×783 практично відтворює оригінал).

  Тобто для тіні такого розміру (значна частка кадру) НЕМАЄ єдиного
  значення kernel_size, при якому один morph-прохід дає прийнятний
  результат. Це і є причина артефакту з чорною плямою/контуром
  телефону — а не баг у конкретних константах.

  Рішення тут: ДРУГИЙ прохід оцінки фону через сильне зменшення
  роздільності (downsample) + GaussianBlur + збільшення назад
  (upsample). При зменшенні до ~32-48px по короткій стороні будь-яка
  тінь (навіть на половину кадру) стає низькочастотним перепадом
  яскравості, і немодельований "стрибок", що створює перший прохід,
  теж зменшується разом з усім зображенням — після upsample він
  розмазується по плавному градієнту, а не лишається різким краєм.
  Цей прохід ВСЕРЕДИНІ не залежить від розміру тіні відносно kernel —
  тому він прибирає саме той клас проблем, де морфологія застрягає.

Обмеження:
- Не застосовується до фото (перевіряється через std каналів A/B та насиченість HSV).
- Не застосовується до документів з гільйошем (паспорти) — детектуються за
  високою насиченістю S-каналу HSV.
- Не застосовується до сильно розмитих зображень (Laplacian var < 10).
- Не застосовується до зображень < 100×100 пікселів (уникнення артефактів).
- Дуже темні й контрастні тіні з різким краєм (тінь від об'єкта, що
  торкається/майже торкається документа, а не просто перекриває
  світло віддалено) прибираються не на 100% — другий прохід сильно
  пом'якшує залишок, але не гарантує ідеально рівний фон. Це
  фундаментальна межа корекції освітлення з одного 2D-зображення:
  немодельовано неможливо відрізнути "що мало бути під тінню" від
  "що там справді надруковано", якщо тінь дуже глибока.

Працює раніше за auto_contrast/CLAHE — тому наступні кроки бачать
"чистий" документ без тіней і не посилюють їх.

Не залежить від жодного іншого модуля проєкту.
"""

import cv2
import numpy as np

# Константи для background estimation (прохід 1 — морфологія)
BLUR_KERNEL_MIN = 21       # мінімальний розмір ядра GaussianBlur (непарне)
BLUR_KERNEL_MAX = 201      # максимальний розмір ядра
BLUR_KERNEL_STEP = 2       # крок збільшення (залишає непарним)
BLUR_SIGMA = 0             # 0 = автоматичний вибір sigma

# Константи для морфологічної обробки
MORPH_KERNEL_MIN = 31      # мінімальний розмір ядра для морфології (більший за Gaussian,
                            # бо MORPH_CLOSE потребує ядра, яке «накриває» всю тінь)
MORPH_SMOOTH_FACTOR = 5    # дільник для обчислення ядра фінального згладжування:
                            # smooth_kernel = max(5, morph_kernel // MORPH_SMOOTH_FACTOR) | 1

# Константи для divide
DIVIDE_SCALE = 255.0       # масштаб для cv2.divide
DIVIDE_EPSILON = 1.0       # мінус до background щоб уникнути ділення на 0

# Константи для автоматичного визначення розміру ядра
KERNEL_SCALE_FACTOR = 5    # kernel = max(min_kernel, min_side // factor)
KERNEL_MIN_SIDE = 100      # мінімальний розмір сторони для масштабування

# Константи для другого ("грубого") проходу — виправляє великі тіні,
# з якими morph_close із проходу 1 принципово не може впоратись
# (див. пояснення у docstring модуля).
COARSE_PASS_ENABLED = True     # глобальний вимикач другого проходу
COARSE_TARGET_SIDE = 48        # короткостороння thumbnail-роздільність для оцінки
                                # широкої тіні; менше значення = сильніша корекція
                                # великих тіней, але вищий ризик "виїдання" градієнтів
                                # самого документа (дуже нерівномірно надрукованих)
COARSE_BLUR_SIGMA = 2.0        # sigma додаткового згладжування thumbnail'у
COARSE_DIVIDE_EPSILON = 1.0    # мінус до background щоб уникнути ділення на 0

# Константи для захисту від артефактів чорних точок
L_MIN_CLAMP = 5                # мінімальне значення L перед діленням щоб уникнути артефактів

# Константи для диференційованої обробки чб vs кольорових документів
COARSE_BLEND_COLOR = 0.0       # сила другого проходу для кольорових (0 = вимкнено)
KERNEL_COLOR_MULTIPLIER = 1.5  # множник ядра першого проходу для кольорових

# Константи для детекції тіней
SHADOW_DETECT_PERCENTILE = 5   # нижній перцентиль для детекції
SHADOW_DETECT_THRESHOLD = 80   # поріг L-каналу: якщо p5 < threshold — є тіні
SHADOW_RATIO_THRESHOLD = 0.3  # мінімальне відношення p5/p95 для визнання тіней

# Константи для захисних механізмів
MIN_IMAGE_SIDE = 100           # мінімальний розмір сторони для обробки
LAPLACIAN_BLUR_THRESHOLD = 10.0 # якщо дисперсія Laplacian < 10 → занадто розмите


def _create_background_model(l_channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Створює модель фону через морфологічне закриття + легке згладжування.

    Алгоритм:
    1. MORPH_CLOSE з еліптичним ядром: dilate заповнює темні області (тіні, текст)
       значеннями світлого фону; erode повертає границю до початкового розміру.
       Еліптичне ядро зберігає геометрію кутів тіней без «сходинок».
    2. GaussianBlur з малим ядром (~kernel_size/5) прибирає залишки текстури
       (дрібний текст, лінії), щоб вони не вплинули на модель фону.

    Працює добре для тіней, ширина яких порівнянна з kernel_size.
    Для тіней, що сильно перевищують kernel_size, дає лише часткове
    покриття (див. docstring модуля) — для цього є другий прохід,
    _create_coarse_background().

    Args:
        l_channel: L-канал LAB зображення (uint8, 2D).
        kernel_size: розмір ядра для морфології (непарне, >= MORPH_KERNEL_MIN).

    Returns:
        Модель фону (uint8, 2D) — згладжене зображення без тіней та текстури.
    """
    # Еліптичне ядро: краще зберігає кути тіней, ніж прямокутне
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Морфологічне закриття: dilate → erode
    # Заповнює темні області (тіні, текст) кольором фону
    closed = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, morph_kernel)

    # Легке згладжування для прибирання залишків текстури
    smooth_size = max(5, kernel_size // MORPH_SMOOTH_FACTOR) | 1
    background = cv2.GaussianBlur(closed, (smooth_size, smooth_size), BLUR_SIGMA)

    return background


def _create_coarse_background(l_channel: np.ndarray) -> np.ndarray:
    """
    Друга, "груба" модель фону: оцінює лише дуже низькочастотну
    (широку) нерівномірність освітлення — саме ту, яку MORPH_CLOSE
    з обмеженим kernel_size не може прибрати, коли тінь більша за ядро.

    Алгоритм:
    1. Зменшуємо L-канал до thumbnail з короткою стороною
       COARSE_TARGET_SIDE пікселів (INTER_AREA — усереднює, тому
       весь дрібний текст/лінії документа зникають самі по собі,
       без окремої морфології).
    2. Легке GaussianBlur thumbnail'у — додатково згладжує.
    3. Збільшуємо назад до оригінального розміру (INTER_CUBIC) —
       інтерполяція дає плавний градієнт без різких меж, тому що
       на такій малій роздільності різких меж вже не існує.

    Через те, що thumbnail настільки малий, будь-яка тінь (навіть
    та, що займає половину кадру) для нього виглядає як низько-
    частотний перепад — той самий метод однаково добре працює і для
    маленької локальної тіні, і для тіні на півкадру, тому що ми
    не прив'язані до фіксованого kernel_size відносно оригінального
    розміру зображення.

    Args:
        l_channel: L-канал LAB зображення (uint8, 2D) — як правило,
            вже після проходу 1 (_create_background_model + divide).

    Returns:
        Груба модель фону (uint8, 2D), того ж розміру що l_channel.
    """
    h, w = l_channel.shape[:2]
    min_side = min(h, w)

    scale = COARSE_TARGET_SIDE / float(min_side)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small = cv2.resize(l_channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigmaX=COARSE_BLUR_SIGMA)
    background = cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_CUBIC)

    return background


def remove_shadow(
    image: np.ndarray,
    kernel_size: int = 0,
    coarse_pass: bool = True,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
) -> np.ndarray:
    """
    Видаляє градієнтні тіні з документа через background estimation.

    Метод (два проходи):
    1. Морфологічне закриття з еліптичним ядром → модель фону без тіней,
       ділимо оригінал на цю модель. Добре прибирає тіні, чий розмір
       порівнянний з обчисленим kernel_size, без ореолів.
    2. (якщо coarse_pass=True) Грубий downsample-based прохід зверху —
       прибирає залишкову широку нерівномірність освітлення, яку
       прохід 1 не міг прибрати через обмеження kernel_size відносно
       розміру тіні (типово: тінь від руки/телефону на половину кадру).

    Args:
        image: BGR numpy array uint8
        kernel_size: Розмір ядра для морфології (0 = автоматичний).
                      Чим більше ядро — тим більші тіні прибирає
                      ПРОХІД 1, але дуже велике ядро починає
                      перетворюватись на no-op (див. docstring модуля).
                      Має бути непарним.
        coarse_pass: Чи застосовувати другий, грубий прохід для
                      великих тіней. За замовчуванням True. Вимкніть,
                      якщо точно знаєте, що тіні завжди дрібні/локальні
                      і хочете заощадити обчислення.
        is_color_document: Якщо True — застосовує множник KERNEL_COLOR_MULTIPLIER
                           до kernel_size і пропускає другий прохід.
        coarse_blend: Сила блендингу другого проходу для кольорових документів.
                      0.0 = другий прохід вимкнено для кольорових.

    Returns:
        Оброблене BGR зображення без градієнтних тіней
    """
    # Захисний механізм: не обробляємо занадто малі зображення
    h, w = image.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        return image.copy()

    if kernel_size == 0:
        kernel_size = _auto_kernel_size(image)

    # Множник ядра для кольорових документів
    if is_color_document:
        kernel_size = int(round(kernel_size * KERNEL_COLOR_MULTIPLIER))

    # Гарантуємо непарність та мінімальний розмір для морфології
    kernel_size = max(kernel_size | 1, MORPH_KERNEL_MIN)

    # Конвертуємо в LAB для роботи з каналом яскравості
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Запобігаємо артефактам чорних точок: L=0 після ділення дає кольоровий піксель
    l_ch = np.maximum(l_ch, L_MIN_CLAMP)

    # --- Прохід 1: морфологічне закриття + легке згладжування ---
    background = _create_background_model(l_ch, kernel_size)

    l_f = l_ch.astype(np.float32)
    bg_f = background.astype(np.float32) + DIVIDE_EPSILON
    l_norm = cv2.divide(l_f, bg_f, scale=DIVIDE_SCALE)
    # Критично: clamp до [0, 255] перед конвертацією, інакше float32 > 255
    # при astype(np.uint8) дають wrapping (mod 256) -> чорна інверсія
    l_norm = np.clip(l_norm, 0.0, 255.0).astype(np.uint8)

    # --- Прохід 2: грубе виправлення залишкової широкої нерівномірності ---
    if coarse_pass and COARSE_PASS_ENABLED:
        # Для кольорових документів: якщо coarse_blend == 0 — пропускаємо другий прохід
        if is_color_document and coarse_blend <= 0.0:
            pass
        else:
            coarse_bg = _create_coarse_background(l_norm)
            l_f2 = l_norm.astype(np.float32)
            coarse_bg_f = coarse_bg.astype(np.float32) + COARSE_DIVIDE_EPSILON
            l_norm2 = cv2.divide(l_f2, coarse_bg_f, scale=DIVIDE_SCALE)
            l_norm_coarse = np.clip(l_norm2, 0.0, 255.0).astype(np.uint8)
            if is_color_document and coarse_blend > 0.0:
                # Блендинг для кольорових: змішуємо результат першого і другого проходу
                l_norm = cv2.addWeighted(l_norm, 1.0 - coarse_blend, l_norm_coarse, coarse_blend, 0)
                l_norm = np.clip(l_norm, 0.0, 255.0).astype(np.uint8)
            else:
                l_norm = l_norm_coarse

    # Збираємо LAB назад
    merged = cv2.merge([l_norm, a_ch, b_ch])
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result


def auto_remove_shadow(
    image: np.ndarray,
    is_color_document: bool = False,
    coarse_blend: float = COARSE_BLEND_COLOR,
    detect_threshold: float = SHADOW_DETECT_THRESHOLD,
    detect_ratio: float = SHADOW_RATIO_THRESHOLD,
) -> tuple[np.ndarray, bool]:
    """
    Автоматичне видалення тіней: спочатку перевіряє чи є тіні,
    потім застосовує remove_shadow якщо потрібно.

    Повертає (результат, чи_були_тіні).
    """
    has_shadow = _detect_shadow(image, threshold=detect_threshold, ratio=detect_ratio)
    if not has_shadow:
        return image.copy(), False

    result = remove_shadow(image, is_color_document=is_color_document, coarse_blend=coarse_blend)
    return result, True


def _auto_kernel_size(image: np.ndarray) -> int:
    """
    Обчислює оптимальний розмір ядра на основі розміру зображення.
    Більше зображення — більше ядро (щоб морфологія «захопила» всю тінь).

    Примітка: це ядро для ПРОХОДУ 1 (морфологія). Воно навмисно
    обмежене BLUR_KERNEL_MAX, бо занадто велике ядро робить
    MORPH_CLOSE по суті no-op'ом (модель фону стає майже константою) —
    компенсацію великих тіней понад цю межу бере на себе прохід 2
    (_create_coarse_background), а не подальше зростання цього ядра.
    """
    h, w = image.shape[:2]
    min_side = min(h, w)

    if min_side < KERNEL_MIN_SIDE:
        return MORPH_KERNEL_MIN

    kernel = min_side // KERNEL_SCALE_FACTOR
    # Робимо непарним та обмежуємо
    kernel = kernel | 1  # гарантуємо непарність
    kernel = max(kernel, MORPH_KERNEL_MIN)
    kernel = min(kernel, BLUR_KERNEL_MAX)

    return kernel


def _detect_shadow(
    image: np.ndarray,
    threshold: float = SHADOW_DETECT_THRESHOLD,
    ratio: float = SHADOW_RATIO_THRESHOLD,
) -> bool:
    """
    Виявляє наявність нерівномірного освітлення (тіней) на зображенні.

    Дві перевірки:
    1. Зображення не надто розмите (Laplacian var >= порогу).
    2. Є темні ділянки з великим перепадом яскравості (p5/95 < ratio).

    Тип документа (фото/кольоровий/чб) визначається зовні і не впливає на
    детекцію тіней — ця функція відповідає тільки на питання "чи є
    нерівне освітлення".
    """
    h, w = image.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        return False

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    # Перевірка 1: зображення не надто розмите
    laplacian_var = float(cv2.Laplacian(l, cv2.CV_64F).var())
    if laplacian_var < LAPLACIAN_BLUR_THRESHOLD:
        return False

    # Перевірка 2: основна — є темні ділянки з великим перепадом яскравості
    p_low = float(np.percentile(l, SHADOW_DETECT_PERCENTILE))
    p_high = float(np.percentile(l, 100 - SHADOW_DETECT_PERCENTILE))

    if p_low < threshold:
        ratio_val = p_low / max(p_high, 1.0)
        return ratio_val < ratio

    return False
