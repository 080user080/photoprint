# PhotoPrint

Програма підготовки фото документів до друку через priPrinter.

## Встановлення

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

## Структура проєкту

```
photoprint/
├── main.py                    # Точка входу
├── requirements.txt
│
├── config/
│   ├── settings.ini           # Текстовий конфіг (редагується блокнотом)
│   └── app_settings.py        # Читання/запис конфігу
│
├── core/
│   ├── loader.py              # Завантаження JPG/PNG/WEBP/TIFF/HEIC
│   ├── saver.py               # Збереження JPG
│   └── printer.py             # Відправка на принтер
│
├── processing/
│   ├── autofix.py             # LAB → CLAHE → Normalize → HDR → Sharpen
│   ├── sharpen.py             # Різкість (Unsharp Mask)
│   ├── hdr.py                 # HDR tone mapping
│   ├── perspective.py         # Авто + ручна перспективна корекція
│   └── pipeline.py            # Єдина точка входу для GUI
│
├── batch/
│   └── batch_processor.py     # Пакетна обробка (авто / ручний режим)
│
├── gui/
│   ├── main_window.py         # Головне вікно
│   ├── preview.py             # Прев'ю До/Після
│   ├── queue_view.py          # Черга файлів з Drag & Drop
│   ├── controls.py            # Слайдери різкості та HDR
│   └── settings_window.py     # Вікно налаштувань
│
└── utils/
    ├── file_utils.py          # Робота з файлами
    └── image_utils.py         # Утиліти numpy/OpenCV
```

## Використання

1. Перетягніть файли у список черги (або Додати файли / Додати папку)
2. Оберіть режим: **Авто** або **Ручний**
3. **Авто** — натисніть "Друкувати все", програма обробить і надрукує без втручання
4. **Ручний** — перегляд кожного фото, Auto Fix або ручне коригування, потім Друк / Пропустити
5. Для перспективи: кнопка "Авто" або "Ручна (4 точки)" → тягніть точки на прев'ю

## Конфіг

Файл `config/settings.ini` — текстовий, редагується блокнотом.
Або через кнопку **⚙ Налаштування** у програмі.
