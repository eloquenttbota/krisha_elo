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


# Медианное расстояние до центра по районам (км) — посчитано из реальных
# координат объявлений в ноутбуке (src/geo.py + krysha_astana_160726.csv).
DISTRICT_DISTANCE = {
    "Есильский р-н":  3.4,
    "Нура р-н":       4.1,
    "р-н Байконур":   4.2,
    "Алматы р-н":     4.4,
    "Сарайшык р-н":   5.3,
    "Сарыарка р-н":   5.9,
}

DISTRICTS = list(DISTRICT_DISTANCE.keys())
HOUSE_TYPES = ["монолитный", "кирпичный", "панельный", "иной"]

(
    AREA, ROOMS, FLOOR, FLOOR_COUNT,
    YEAR, CEILING, DISTRICT, HOUSE_TYPE, OWNER, IN_COMPLEX,
) = range(10)

# Вопросы и подсказки для каждого шага
PROMPTS = {
    AREA:        "📐 *Площадь квартиры* — в м²\n_от 15 до 300_",
    ROOMS:       "🚪 *Количество комнат*\n_от 1 до 7_",
    FLOOR:       "🏢 *Этаж квартиры*\n_от 1 до 40_",
    FLOOR_COUNT: "🏗 *Этажей в доме*\n_от 2 до 40_",
    YEAR:        "📅 *Год постройки*\n_от 1960 до 2028_",
    CEILING:     "📏 *Высота потолков* — в метрах\n_от 2.3 до 4.2_",
}

ERRORS = {
    AREA:        "⚠️ Введите число от *15 до 300*\n_например: 65 или 65.5_",
    ROOMS:       "⚠️ Введите число от *1 до 7*\n_например: 3_",
    FLOOR:       "⚠️ Введите число от *1 до 40*\n_например: 5_",
    FLOOR_COUNT: "⚠️ Этажей должно быть от *{floor} до 40*\n_например: 9_",
    YEAR:        "⚠️ Введите год от *1960 до 2028*\n_например: 2010_",
    CEILING:     "⚠️ Высота от *2.3 до 4.2* метров\n_например: 2.7_",
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
    IN_COMPLEX:  OWNER,
}


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("‹ Назад", callback_data="back")]])


def choice_keyboard(options: list[str], with_back: bool = True, columns: int = 2) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(options[i + j], callback_data=options[i + j]) for j in range(columns) if i + j < len(options)]
        for i in range(0, len(options), columns)
    ]
    if with_back:
        rows.append([InlineKeyboardButton("‹ Назад", callback_data="back")])
    return InlineKeyboardMarkup(rows)


def restart_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Оценить другую квартиру", callback_data="restart")]])


async def ask(update_or_query, text: str, keyboard=None, parse_mode="Markdown"):
    """Отправляет или редактирует сообщение."""
    if hasattr(update_or_query, "message") and update_or_query.message:
        await update_or_query.message.reply_text(text, reply_markup=keyboard, parse_mode=parse_mode)
    else:
        await update_or_query.edit_message_text(text, reply_markup=keyboard, parse_mode=parse_mode)


# ─── Обработчики шагов ────────────────────────────────────────────

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.clear()
    text = "🏠 *Оценка квартиры в Астане*\n\nОтвечайте на вопросы — я рассчитаю стоимость.\n\n" + PROMPTS[AREA]
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(text, parse_mode="Markdown")
    else:
        await update.message.reply_text(text, parse_mode="Markdown")
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
        assert 15 <= val <= 300
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
        assert 1 <= val <= 7
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
        assert 1 <= val <= 40
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
        assert floor <= val <= 40
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
        assert 1960 <= val <= 2028
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
        assert 2.3 <= val <= 4.2
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
    ctx.user_data["district"] = val
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
    ctx.user_data["house_type"] = val
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
    ctx.user_data["_state"] = IN_COMPLEX
    await query.edit_message_text(
        f"Продавец: *{query.data}*\n\nКвартира в жилом комплексе (ЖК)?",
        reply_markup=choice_keyboard(["Да", "Нет"]),
        parse_mode="Markdown",
    )
    return IN_COMPLEX


async def get_in_complex(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "back":
        return await handle_back(update, ctx)
    ctx.user_data["in_complex"] = query.data == "Да"
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
        district = ctx.user_data.get("district", "—")
        await query.message.reply_text(
            f"✅ *Результат оценки*\n"
            f"{'─' * 22}\n"
            f"📍 Район: {district}\n"
            f"📐 Площадь: {area} м²\n"
            f"{'─' * 22}\n"
            f"💰 Цена за м²: *{price_m2} ₸*\n"
            f"🏠 Итого: *{total} ₸*",
            parse_mode="Markdown",
            reply_markup=restart_keyboard(),
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
        entry_points=[
            CommandHandler("start", start),
            CallbackQueryHandler(start, pattern="^restart$"),
        ],
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
            IN_COMPLEX:  [CallbackQueryHandler(get_in_complex)],
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
