import asyncio
import os
import uvicorn
from backend.main import app
from bot.bot import build_app


async def main():
    port = int(os.getenv("PORT", 8000))

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)

    bot_app = build_app()

    async with bot_app:
        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling()

        await server.serve()

        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
