"""
1. timed — decorator dùng được cho cả hàm thường lẫn hàm async.
Ghi log ở mức INFO qua logger multi_caller, theo dạng "<tên hàm> took <x.xxx>s".
Vẫn ghi log khi hàm raise lỗi.
Giữ nguyên metadata của hàm gốc.
"""
import functools
import inspect
import json
import logging
import time

log = logging.getLogger("multi_caller")

def timed(fn):
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                log.info("%s took %.3fs", fn.__name__, time.perf_counter() - start)
        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try: 
            return fn(*args, **kwargs)
        finally:
            log.info("%s took %.3fs", fn.__name__, time.perf_counter() - start)
    return wrapper


"""
2. read_jsonl(path) — generator lazy.
Bỏ qua dòng trống.
Bỏ qua dòng JSON hỏng và ghi log WARNING.
"""
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skip bad line %d in %s", lineno, path)
                                      


"""
3. batched(iterable, n) — generator trả về các list dài tối đa n phần tử.
Không dùng itertools.batched.
n < 1 thì raise ValueError.
"""
def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be >=1")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
