"""Cho sẵn — không cần sửa. Các kiểu dữ liệu dùng chung của project."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str
    price_in: float
    price_out: float


@dataclass
class CallResult:
    provider: str
    prompt_id: int
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    attempts: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
