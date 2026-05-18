import os
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

load_dotenv(Path(__file__).parent.parent / ".env")

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def get_api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8000")


def clean_float(text: str) -> float:
    """65,5 / 65.5 / '65 5' / '65 м²' → 65.5"""
    import re
    t = text.strip()
    t = re.sub(r"[^\d.,\-]", "", t)       # оставляем только цифры, . , -
    t = re.sub(r"[,\s](?=\d{3}(?:[,\s]|$))", "", t)  # убираем разделители тысяч: 1 500 → 1500
    t = t.replace(",", ".")               # европейский разделитель: 65,5 → 65.5
    return float(t)


def clean_int(text: str) -> int:
    import re
    t = text.strip()
    t = re.sub(r"[^\d]", "", t)           # только цифры
    return int(t)


# Приблизительное расстояние до центра по районам (км)
DISTRICT_DISTANCE = {
    "Есильский р-н":  3.5,
    "Нура р-н":       10.0,
    "Сарайшык р-н":   12.0,
    "Сарыарка р-н":    6.0,
    "р-н Байконур":    8.0,
    "Другой":          7.0,
}

DISTRICTS = list(DISTRICT_DISTANCE.keys())
HOUSE_TYPES = ["монолитный дом", "панельный дом", "кирпичный дом", "другой"]

(
    AREA, ROOMS, FLOOR, FLOOR_COUNT,
    YEAR, CEILING, DISTRICT, HOUSE_TYPE, OWNER,
) = range(9)

# Вопросы и подсказки для каждого шага
PROMPTS = {
    AREA:        "Введите площадь квартиры в м²:\n_Пример: 65 или 65.5 (от 10 до 500 м²)_",
    ROOMS:       "Количество комнат:\n_Пример: 2 (от 1 до 10)_",
    FLOOR:       "На каком этаже квартира?\n_Пример: 5 (от 1 до 100)_",
    FLOOR_COUNT: "Сколько этажей в доме?\n_Пример: 9 (от 1 до 100)_",
    YEAR:        "Год постройки дома:\n_Пример: 2010 (от 1950 до 2025)_",
    CEILING:     "Высота потолков в метрах:\n_Пример: 2.7 (от 1.5 до 6.0)_",
}

ERRORS = {
    AREA:        "⚠️ Площадь должна быть числом от *10 до 500* м²\nПример: *65* или *65.5*",
    ROOMS:       "⚠️ Введите целое число от *1 до 10*\nПример: *2*",
    FLOOR:       "⚠️ Введите целое число от *1 до 100*\nПример: *5*",
    FLOOR_COUNT: "⚠️ Этажей в доме должно быть от *{floor} до 100*\nПример: *9*",
    YEAR:        "⚠️ Введите год от *1950 до 2025*\nПример: *2010*",
    CEILING:     "⚠️ Высота потолков от *1.5 до 6.0* метров\nПример: *2.7*",
}

PREV_STATE = {
    ROOMS:       AREA,
    FLOOR:       ROOMS,
    FLOOR_COUNT: FLOOR,
    YEAR:        FLOOR_COUNT,
    CEILING:     YEAR,
    DISTRICT:    CEILING,
    HOUSE_TYPE:  DISTRICT,
    OWNER:       HOUSE_TYPE,
}


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back")]])


def choice_keyboard(options: list[str], with_back: bool = True) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(o, callback_data=o)] for o in options]
    if with_back:
        buttons.append([InlineKeyboardButton("← Назад", callback_data="back")])
    return InlineKeyboardMarkup(buttons)


async def ask(update_or_query, text: str, keyboard=None, parse_mode="Markdown"):
    """Отправляет или редактирует сообщение."""
    if hasattr(update_or_query, "message") and update_or_query.message:
        await update_or_query.message.reply_text(text, reply_markup=keyboard, parse_mode=parse_mode)
    else:
        await update_or_query.edit_message_text(text, reply_markup=keyboard, parse_mode=parse_mode)


# ─── Обработчики шагов ────────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.clear()
    await update.message.reply_text(
        "Привет! Я предскажу цену квартиры в Астане 🏠\n\n" + PROMPTS[AREA],
        parse_mode="Markdown",
    )
    return AREA


async def handle_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    current = ctx.user_data.get("_state", AREA)
    prev = PREV_STATE.get(current, AREA)
    ctx.user_data["_state"] = prev

    if prev in PROMPTS:
        await query.edit_message_text(PROMPTS[prev], reply_markup=back_keyboard(), parse_mode="Markdown")
    elif prev == DISTRICT:
        await query.edit_message_text("Выберите район:", reply_markup=choice_keyboard(DISTRICTS, with_back=False), parse_mode="Markdown")
    elif prev == HOUSE_TYPE:
        await query.edit_message_text("Выберите тип дома:", reply_markup=choice_keyboard(HOUSE_TYPES), parse_mode="Markdown")
    elif prev == OWNER:
        await query.edit_message_text("Кто продаёт квартиру?", reply_markup=choice_keyboard(["Хозяин", "Агентство"]), parse_mode="Markdown")
    return prev


async def get_area(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = AREA
    try:
        val = clean_float(update.message.text)
        assert 10 <= val <= 500
        ctx.user_data["area"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(ERRORS[AREA], reply_markup=back_keyboard(), parse_mode="Markdown")
        return AREA
    ctx.user_data["_state"] = ROOMS
    await update.message.reply_text(PROMPTS[ROOMS], reply_markup=back_keyboard(), parse_mode="Markdown")
    return ROOMS


async def get_rooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = ROOMS
    try:
        val = clean_int(update.message.text)
        assert 1 <= val <= 10
        ctx.user_data["room_count"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(ERRORS[ROOMS], reply_markup=back_keyboard(), parse_mode="Markdown")
        return ROOMS
    ctx.user_data["_state"] = FLOOR
    await update.message.reply_text(PROMPTS[FLOOR], reply_markup=back_keyboard(), parse_mode="Markdown")
    return FLOOR


async def get_floor(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = FLOOR
    try:
        val = clean_int(update.message.text)
        assert 1 <= val <= 100
        ctx.user_data["floor"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(ERRORS[FLOOR], reply_markup=back_keyboard(), parse_mode="Markdown")
        return FLOOR
    ctx.user_data["_state"] = FLOOR_COUNT
    await update.message.reply_text(PROMPTS[FLOOR_COUNT], reply_markup=back_keyboard(), parse_mode="Markdown")
    return FLOOR_COUNT


async def get_floor_count(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = FLOOR_COUNT
    floor = ctx.user_data.get("floor", 1)
    try:
        val = clean_int(update.message.text)
        assert floor <= val <= 100
        ctx.user_data["floor_count"] = val
    except (ValueError, AssertionError):
        msg = ERRORS[FLOOR_COUNT].format(floor=floor)
        await update.message.reply_text(msg, reply_markup=back_keyboard(), parse_mode="Markdown")
        return FLOOR_COUNT
    ctx.user_data["_state"] = YEAR
    await update.message.reply_text(PROMPTS[YEAR], reply_markup=back_keyboard(), parse_mode="Markdown")
    return YEAR


async def get_year(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = YEAR
    try:
        val = clean_int(update.message.text)
        assert 1950 <= val <= 2025
        ctx.user_data["construction_year"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(ERRORS[YEAR], reply_markup=back_keyboard(), parse_mode="Markdown")
        return YEAR
    ctx.user_data["_state"] = CEILING
    await update.message.reply_text(PROMPTS[CEILING], reply_markup=back_keyboard(), parse_mode="Markdown")
    return CEILING


async def get_ceiling(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data["_state"] = CEILING
    try:
        val = clean_float(update.message.text)
        assert 1.5 <= val <= 6.0
        ctx.user_data["ceiling_height"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(ERRORS[CEILING], reply_markup=back_keyboard(), parse_mode="Markdown")
        return CEILING
    ctx.user_data["_state"] = DISTRICT
    await update.message.reply_text(
        "Выберите район:",
        reply_markup=choice_keyboard(DISTRICTS, with_back=False),
        parse_mode="Markdown",
    )
    return DISTRICT


async def get_district(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        return await handle_back(update, ctx)
    val = query.data
    ctx.user_data["district"] = "" if val == "Другой" else val
    ctx.user_data["distance_to_center"] = DISTRICT_DISTANCE[val]
    ctx.user_data["_state"] = HOUSE_TYPE
    await query.edit_message_text(
        f"Район: *{val}*\n\nВыберите тип дома:",
        reply_markup=choice_keyboard(HOUSE_TYPES),
        parse_mode="Markdown",
    )
    return HOUSE_TYPE


async def get_house_type(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        return await handle_back(update, ctx)
    val = query.data
    ctx.user_data["house_type"] = "" if val == "другой" else val
    ctx.user_data["_state"] = OWNER
    await query.edit_message_text(
        f"Тип дома: *{val}*\n\nКто продаёт квартиру?",
        reply_markup=choice_keyboard(["Хозяин", "Агентство"]),
        parse_mode="Markdown",
    )
    return OWNER


async def get_owner(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        return await handle_back(update, ctx)
    ctx.user_data["owner"] = query.data
    await query.edit_message_text("Считаю цену... ⏳")

    payload = {k: v for k, v in ctx.user_data.items() if not k.startswith("_")}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{get_api_url()}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        price_m2 = f"{result['price_per_m2']:,}".replace(",", " ")
        total = f"{result['total_price']:,}".replace(",", " ")
        area = ctx.user_data.get("area", 0)
        await query.message.reply_text(
            f"*Прогноз цены квартиры:*\n\n"
            f"📐 Площадь: {area} м²\n"
            f"💰 Цена за м²: *{price_m2} ₸*\n"
            f"🏠 Общая стоимость: *{total} ₸*\n\n"
            f"Чтобы оценить другую квартиру — /start",
            parse_mode="Markdown",
        )
    except Exception as e:
        logging.error("Predict error: %s", e)
        await query.message.reply_text(
            f"⚠️ Ошибка при расчёте. Попробуйте ещё раз: /start",
            parse_mode="Markdown",
        )
    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отменено. Для нового расчёта — /start")
    return ConversationHandler.END


async def error_handler(_update, context):
    logging.error("Ошибка: %s", context.error, exc_info=context.error)


def build_app() -> Application:
    bot_app = Application.builder().token(TOKEN).build()

    back_filter = filters.TEXT & ~filters.COMMAND

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AREA:        [MessageHandler(back_filter, get_area),        CallbackQueryHandler(handle_back, pattern="^back$")],
            ROOMS:       [MessageHandler(back_filter, get_rooms),       CallbackQueryHandler(handle_back, pattern="^back$")],
            FLOOR:       [MessageHandler(back_filter, get_floor),       CallbackQueryHandler(handle_back, pattern="^back$")],
            FLOOR_COUNT: [MessageHandler(back_filter, get_floor_count), CallbackQueryHandler(handle_back, pattern="^back$")],
            YEAR:        [MessageHandler(back_filter, get_year),        CallbackQueryHandler(handle_back, pattern="^back$")],
            CEILING:     [MessageHandler(back_filter, get_ceiling),     CallbackQueryHandler(handle_back, pattern="^back$")],
            DISTRICT:    [CallbackQueryHandler(get_district)],
            HOUSE_TYPE:  [CallbackQueryHandler(get_house_type)],
            OWNER:       [CallbackQueryHandler(get_owner)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
    )

    bot_app.add_handler(conv)
    bot_app.add_error_handler(error_handler)
    return bot_app


def main():
    bot_app = build_app()
    print("Бот запущен...")
    bot_app.run_polling()


if __name__ == "__main__":
    main()
