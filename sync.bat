@echo off
cd /d "d:\Python\Print_COPY\photoprint\"

echo Checking for changes...

:: Додає ВСЕ: нові, змінені і ВИДАЛЕНІ файли
git add -A

:: Коміт тільки якщо є зміни
set msg=Update: %date% %time%
git commit -m "%msg%"

:: Примусовий пуш (опціонально, але корисно)
git push origin main

echo.
echo Sync complete!
timeout /t 5