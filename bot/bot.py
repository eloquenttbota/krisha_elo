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

(
    AREA, ROOMS, FLOOR, FLOOR_COUNT,
    YEAR, CEILING, DISTANCE,
    DISTRICT, HOUSE_TYPE, OWNER,
) = range(10)

DISTRICTS = [
    "Есильский р-н", "Нура р-н", "Сарайшык р-н",
    "Сарыарка р-н", "р-н Байконур", "Другой",
]
HOUSE_TYPES = ["монолитный дом", "панельный дом", "другой"]


def make_keyboard(options: list[str]) -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(o, callback_data=o)] for o in options]
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.clear()
    await update.message.reply_text(
        "Привет! Я предскажу цену квартиры в Астане.\n\n"
        "Введите площадь квартиры в м²:"
    )
    return AREA


async def get_area(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        ctx.user_data["area"] = float(update.message.text.replace(",", "."))
    except ValueError:
        await update.message.reply_text("Введите число, например: 65.5")
        return AREA
    await update.message.reply_text("Количество комнат (1–10):")
    return ROOMS


async def get_rooms(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = int(update.message.text)
        assert 1 <= val <= 10
        ctx.user_data["room_count"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text("Введите целое число от 1 до 10:")
        return ROOMS
    await update.message.reply_text("На каком этаже квартира?")
    return FLOOR


async def get_floor(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        ctx.user_data["floor"] = int(update.message.text)
    except ValueError:
        await update.message.reply_text("Введите целое число:")
        return FLOOR
    await update.message.reply_text("Сколько этажей в доме?")
    return FLOOR_COUNT


async def get_floor_count(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = int(update.message.text)
        assert val >= ctx.user_data["floor"]
        ctx.user_data["floor_count"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text(
            f"Этажей в доме должно быть не меньше {ctx.user_data['floor']}. Введите снова:"
        )
        return FLOOR_COUNT
    await update.message.reply_text("Год постройки дома (например: 2005):")
    return YEAR


async def get_year(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = int(update.message.text)
        assert 1950 <= val <= 2025
        ctx.user_data["construction_year"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text("Введите год от 1950 до 2025:")
        return YEAR
    await update.message.reply_text("Высота потолков в метрах (например: 2.7):")
    return CEILING


async def get_ceiling(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = float(update.message.text.replace(",", "."))
        assert 1.5 <= val <= 6.0
        ctx.user_data["ceiling_height"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text("Введите число от 1.5 до 6.0:")
        return CEILING
    await update.message.reply_text("Расстояние до центра города (в км, например: 3.5):")
    return DISTANCE


async def get_distance(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = float(update.message.text.replace(",", "."))
        assert val > 0
        ctx.user_data["distance_to_center"] = val
    except (ValueError, AssertionError):
        await update.message.reply_text("Введите положительное число:")
        return DISTANCE
    await update.message.reply_text(
        "Выберите район:", reply_markup=make_keyboard(DISTRICTS)
    )
    return DISTRICT


async def get_district(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    val = query.data
    ctx.user_data["district"] = "" if val == "Другой" else val
    await query.edit_message_text(
        f"Район: {val}\n\nВыберите тип дома:",
        reply_markup=make_keyboard(HOUSE_TYPES),
    )
    return HOUSE_TYPE


async def get_house_type(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    val = query.data
    ctx.user_data["house_type"] = "" if val == "другой" else val
    await query.edit_message_text(
        f"Тип дома: {val}\n\nКто продаёт квартиру?",
        reply_markup=make_keyboard(["Хозяин", "Агентство"]),
    )
    return OWNER


async def get_owner(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    ctx.user_data["owner"] = query.data
    await query.edit_message_text("Считаю цену...")

    payload = ctx.user_data.copy()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{get_api_url()}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        price_m2 = f"{result['price_per_m2']:,}".replace(",", " ")
        total = f"{result['total_price']:,}".replace(",", " ")
        await query.message.reply_text(
            f"Прогноз цены:\n\n"
            f"Цена за м²: {price_m2} ₸\n"
            f"Общая стоимость: {total} ₸\n\n"
            f"Чтобы оценить другую квартиру — /start"
        )
    except Exception as e:
        await query.message.reply_text(
            f"Ошибка при расчёте: {e}\n\nПопробуйте ещё раз: /start"
        )
    return ConversationHandler.END


async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Отменено. Для нового расчёта — /start")
    return ConversationHandler.END


def build_app() -> Application:
    bot_app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            ROOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_rooms)],
            FLOOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor)],
            FLOOR_COUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor_count)],
            YEAR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_year)],
            CEILING: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_ceiling)],
            DISTANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_distance)],
            DISTRICT: [CallbackQueryHandler(get_district)],
            HOUSE_TYPE: [CallbackQueryHandler(get_house_type)],
            OWNER: [CallbackQueryHandler(get_owner)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    async def error_handler(update, context):
        logging.error("Ошибка: %s", context.error, exc_info=context.error)

    bot_app.add_handler(conv)
    bot_app.add_error_handler(error_handler)
    return bot_app


def main():
    bot_app = build_app()
    print("Бот запущен...")
    bot_app.run_polling()


if __name__ == "__main__":
    main()
