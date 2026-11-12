import asyncio
import os
from pathlib import Path
from typing import Literal

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Control Center", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=12000)
    mode: Literal["parallel", "review", "sequential"] = "review"


class ChatResponse(BaseModel):
    mode: str
    openai: str | None = None
    claude: str | None = None
    final: str


def check_password(candidate: str | None) -> None:
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Сервис ещё не настроен: задайте APP_PASSWORD в переменных окружения.",
        )
    if candidate != expected:
        raise HTTPException(status_code=401, detail="Неверный пароль приложения.")


def get_openai_client() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Не настроен OPENAI_API_KEY.")
    return AsyncOpenAI(api_key=key, timeout=90.0, max_retries=1)


def get_anthropic_client() -> AsyncAnthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Не настроен ANTHROPIC_API_KEY.")
    return AsyncAnthropic(api_key=key, timeout=90.0, max_retries=1)


async def ask_openai(instructions: str, prompt: str) -> str:
    client = get_openai_client()
    response = await client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        instructions=instructions,
        input=prompt,
        max_output_tokens=2500,
    )
    return response.output_text.strip()


async def ask_claude(system: str, prompt: str) -> str:
    client = get_anthropic_client()
    response = await client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
        system=system,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "claude_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "app_password_configured": bool(os.getenv("APP_PASSWORD")),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_app_password: str | None = Header(default=None),
):
    check_password(x_app_password)

    # Fail early with clear messages rather than silently skipping an agent.
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="Добавьте OPENAI_API_KEY в Render.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="Сначала добавьте ANTHROPIC_API_KEY в Render.")

    coordinator = (
        "Ты — координатор AI Control Center. Отвечай по-русски, отделяй факты от "
        "предположений, не выдумывай источники и цифры. Если данных недостаточно, "
        "прямо укажи ограничения. Сохраняй практическую структуру ответа."
    )
    analyst = (
        "Ты — независимый эксперт-аналитик. Дай самостоятельный, аргументированный "
        "ответ по-русски. Указывай допущения, риски и то, что требует проверки."
    )
    critic = (
        "Ты — строгий эксперт-рецензент. Ищи логические ошибки, недоказанные тезисы, "
        "пропущенные риски и практические недостатки. Не соглашайся автоматически. "
        "Отвечай по-русски."
    )

    try:
        if request.mode == "parallel":
            gpt_result, claude_result = await asyncio.gather(
                ask_openai(analyst, request.message),
                ask_claude(analyst, request.message),
            )
            final = await ask_openai(
                coordinator,
                "Сопоставь два независимых ответа. Отметь совпадения и разногласия, "
                "не скрывай неопределённость. Сформируй итог.\n\n"
                f"ЗАДАЧА:\n{request.message}\n\n"
                f"АНАЛИЗ CHATGPT:\n{gpt_result}\n\n"
                f"АНАЛИЗ CLAUDE:\n{claude_result}",
            )
        elif request.mode == "sequential":
            gpt_result = await ask_openai(analyst, request.message)
            claude_result = await ask_claude(
                analyst,
                f"Улучши и дополни этот первичный анализ. Исправляй ошибки, "
                f"обозначай сомнительные места.\n\nЗАДАЧА:\n{request.message}\n\n"
                f"ЧЕРНОВИК:\n{gpt_result}",
            )
            final = await ask_openai(
                coordinator,
                f"Подготовь окончательный ответ с учётом рецензии Claude.\n\n"
                f"ЗАДАЧА:\n{request.message}\n\nЧЕРНОВИК:\n{gpt_result}\n\n"
                f"УЛУЧШЕНИЯ CLAUDE:\n{claude_result}",
            )
        else:  # review
            gpt_result = await ask_openai(analyst, request.message)
            claude_result = await ask_claude(
                critic,
                f"Проведи критическую проверку ответа ChatGPT. Перечисли сильные "
                f"стороны, ошибки, недоказанные утверждения и конкретные улучшения.\n\n"
                f"ИСХОДНАЯ ЗАДАЧА:\n{request.message}\n\nОТВЕТ:\n{gpt_result}",
            )
            final = await ask_openai(
                coordinator,
                f"Сформируй улучшенный окончательный ответ. Учти обоснованную критику, "
                f"но не принимай её без проверки. Укажи нерешённые разногласия.\n\n"
                f"ЗАДАЧА:\n{request.message}\n\nЧЕРНОВИК CHATGPT:\n{gpt_result}\n\n"
                f"РЕЦЕНЗИЯ CLAUDE:\n{claude_result}",
            )

        return ChatResponse(
            mode=request.mode,
            openai=gpt_result,
            claude=claude_result,
            final=final,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose provider response bodies, credentials, or stack traces.
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка обращения к AI-провайдеру ({type(exc).__name__}). "
                   "Проверьте API-ключи, доступ к модели и лимиты аккаунта.",
        ) from None
