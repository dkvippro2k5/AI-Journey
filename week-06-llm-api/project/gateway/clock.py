"""Đồng hồ (cho sẵn, KHÔNG cần sửa).

Mọi chỗ cần đo thời gian hoặc chờ trong gateway đều đi qua một đối tượng `clock` có hai phương thức:
`now() -> float` (giây) và `async sleep(dt)`.

- `RealClock`: thời gian thật (`time.perf_counter`, `asyncio.sleep`). Dùng khi gọi provider thật.
- `SimClock`: thời gian mô phỏng. `sleep(dt)` chỉ cộng `dt` vào đồng hồ rồi trả ngay, nên test và benchmark chạy
  trong vài giây dù "độ trễ" mô phỏng là hàng giờ, và kết quả lặp lại được. Chỉ đúng khi các request được xử lý
  TUẦN TỰ (benchmark làm vậy: mỗi request có dòng thời gian riêng, bắt đầu từ lúc nó đến).
"""
from __future__ import annotations

import asyncio
import time


class RealClock:
    def now(self) -> float:
        return time.perf_counter()

    async def sleep(self, dt: float) -> None:
        await asyncio.sleep(max(0.0, dt))


class SimClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def now(self) -> float:
        return self.t

    async def sleep(self, dt: float) -> None:
        if dt > 0:
            self.t += dt
        await asyncio.sleep(0)                      # nhường event loop như một lần chờ thật

    def set(self, t: float) -> None:
        """Đặt lại đồng hồ (benchmark gọi trước mỗi request: t = thời điểm request đến)."""
        self.t = float(t)
