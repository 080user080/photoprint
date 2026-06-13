@echo off
chcp 65001 >nul
cd /d "d:\Python\Print_COPY\photoprint\photoprint-main"

:: Перевірка чи є .git (на випадок якщо папка .git видалена)
if not exist ".git" (
    echo Initializing git repository...
    git init
    git remote add origin https://github.com/080user080/photoprint.git
)

echo.
echo === Syncing to GitHub ===
echo.

:: Додає ВСЕ: нові, змінені і ВИДАЛЕНІ файли
git add -A

:: Коміт тільки якщо є зміни
set msg=Update: %date% %time%
git commit -m "%msg%"

:: Пуш в main
git push origin main

echo.
echo === Sync complete! ===
echo.
timeout /t 5