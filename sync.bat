@echo off
:: Перехід у папку проекту
cd /d "d:\Python\Print_COPY\photoprint\"

echo Checking for changes...

:: 1. Додаємо всі нові та змінені файли (враховуючи .gitignore)
git add .

:: 2. Створюємо коміт лише якщо є зміни
:: Повідомлення міститиме дату та час для зручності
set msg=Update: %date% %time%
git commit -m "%msg%"

:: 3. Відправляємо зміни у вже існуючу гілку main на GitHub
:: "origin" — це назва вашого віддаленого репозиторію, "main" — назва гілки
git push origin main

echo.
echo Sync complete!
timeout /t 5