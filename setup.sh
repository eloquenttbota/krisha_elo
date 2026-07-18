#!/usr/bin/env bash
# Готовит рабочее виртуальное окружение с нуля.
# Использование:
#   ./setup.sh          — окружение для бота/бэкенда (requirements.txt)
#   ./setup.sh --dev     — + инструменты для ноутбука (requirements-dev.txt)
#
# ВАЖНО: venv/ нельзя переносить между компьютерами (архивом/копированием) —
# он содержит абсолютные пути и собранные под конкретную машину бинарники.
# При переезде на новый компьютер удалите venv/ и запустите этот скрипт заново.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

REQ_FILE="requirements.txt"
if [[ "${1:-}" == "--dev" ]]; then
    REQ_FILE="requirements-dev.txt"
fi

echo "→ Проверяю системные зависимости..."
if [[ "$(uname)" == "Darwin" ]]; then
    # xgboost на macOS требует OpenMP runtime, который pip не ставит
    if ! (brew list libomp >/dev/null 2>&1); then
        if command -v brew >/dev/null 2>&1; then
            echo "→ Устанавливаю libomp (нужен для xgboost на macOS)..."
            brew install libomp
        else
            echo "⚠️  Homebrew не найден. Установите libomp вручную: https://brew.sh, затем 'brew install libomp'"
            echo "   Без этого шага 'import xgboost' упадёт с ошибкой libomp.dylib."
        fi
    else
        echo "  libomp уже установлен."
    fi
fi

echo "→ Пересоздаю виртуальное окружение (venv/)..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

echo "→ Обновляю pip..."
python3 -m pip install --upgrade pip --quiet

echo "→ Устанавливаю зависимости из ${REQ_FILE}..."
pip install -r "${REQ_FILE}"

echo "→ Проверяю, что всё импортируется..."
python3 - <<'PYEOF'
import importlib
mods = ["numpy", "pandas", "sklearn", "xgboost", "joblib",
        "fastapi", "uvicorn", "httpx", "dotenv", "telegram"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append((m, str(e)))
if failed:
    print("\n❌ Не удалось импортировать:")
    for m, err in failed:
        print(f"  {m}: {err}")
    raise SystemExit(1)
print("✅ Все ключевые библиотеки импортируются успешно.")
PYEOF

echo ""
echo "Готово. Активируйте окружение командой:"
echo "  source venv/bin/activate"
