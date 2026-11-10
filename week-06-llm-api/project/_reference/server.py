"""Ngày 4 · Backend proxy bằng FastAPI: trình duyệt/app gọi server của bạn, server gọi provider.
API key của provider chỉ nằm trong biến môi trường của server; khách xác thực bằng token riêng của hệ thống.

    uvicorn gateway.server:app        # cần GATEWAY_DEMO=1 để tạo app demo với FakeLLM (xem cuối file)
"""
from __future__ import annotations

import asyncio
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gateway.stream import sse_encode


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")              # khách không được tự chèn message "system"
    content: str = Field(max_length=2000)


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    history: list[Msg] = Field(default_factory=list, max_length=20)


def public(res) -> dict:
    """Chỉ trả về những gì khách cần; không lộ model, chi phí, lỗi nội bộ."""
    return {"reply": res.reply, "intent": res.intent, "route": res.route}


def create_app(gateway, user_tokens: dict[str, str]) -> FastAPI:
    """`user_tokens`: token của khách → user_id (ngoài đời là phiên đăng nhập / JWT, không phải API key của provider)."""
    app = FastAPI(title="LLM gateway")

    def current_user(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        if token not in user_tokens:
            raise HTTPException(401, "invalid token")
        return user_tokens[token]

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.post("/v1/chat")
    async def chat(body: ChatIn, user: str = Depends(current_user)):
        res = await gateway.handle(user, body.text, [m.model_dump() for m in body.history])
        if res.route == "rate_limited":
            wait = gateway.limiter.retry_after(user) if gateway.limiter else 1.0
            return JSONResponse(public(res), status_code=429, headers={"Retry-After": str(max(1, round(wait)))})
        return public(res)

    @app.post("/v1/chat/stream")
    async def chat_stream(body: ChatIn, user: str = Depends(current_user)):
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(gateway.handle(user, body.text, [m.model_dump() for m in body.history],
                                                  on_reply_delta=queue.put_nowait))

        async def events():
            sent = []
            while not (task.done() and queue.empty()):
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    sent.append(getter.result())
                    yield sse_encode({"delta": sent[-1]}, event="delta")
                else:
                    getter.cancel()
            res = task.result()
            final = public(res)
            final["replace"] = "".join(sent) != res.reply                # phần đã stream khác câu trả lời cuối
            yield sse_encode(final, event="done")

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def demo_app() -> FastAPI:
    """App demo chạy offline với FakeLLM (để thử bằng uvicorn). Token khách demo: "demo-token"."""
    from gateway.app import Gateway
    from gateway.client import LLMClient
    from gateway.data import make_workload
    from gateway.fake import FakeLLM, build_oracle
    from gateway.router import default_router
    fake = FakeLLM(oracle=build_oracle(make_workload(2000, 0)))
    client = LLMClient(fake.client(), api_key=os.environ.get("FAKE_API_KEY", "test-key"))
    return create_app(Gateway(client, default_router()), {"demo-token": "u-demo"})


app = demo_app() if os.environ.get("GATEWAY_DEMO") == "1" else None
