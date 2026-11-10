from gateway.context import fit_history, summarize_old, truncate_middle
from gateway.tokens import count_chat_tokens, count_tokens


def _conv(n_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "Bạn là trợ lý tổng đài."}]
    for i in range(n_turns):
        msgs += [{"role": "user", "content": f"câu hỏi số {i} về gói cước và data còn lại"},
                 {"role": "assistant", "content": f"câu trả lời số {i}: anh chị soạn KT gửi 1234"}]
    return msgs + [{"role": "user", "content": "vậy còn gói GX30 thì sao?"}]


def test_fit_history_keeps_system_and_last_and_order():
    msgs = _conv(10)
    full = count_chat_tokens(msgs)
    out, dropped = fit_history(msgs, budget=full // 2)
    assert count_chat_tokens(out) <= full // 2
    assert out[0] == msgs[0] and out[-1] == msgs[-1]
    assert out[1]["role"] == "user"                                               # lịch sử không bắt đầu bằng assistant
    assert out[1:] == msgs[len(msgs) - len(out) + 1:]                             # giữ đúng phần đuôi, đúng thứ tự
    assert dropped == len(msgs) - len(out)
    assert fit_history(msgs, budget=10 ** 6) == (msgs, 0)


def test_fit_history_when_nothing_fits():
    msgs = _conv(3)
    out, dropped = fit_history(msgs, budget=5)
    assert out == [msgs[0], msgs[-1]] and dropped == 6
    custom, n = fit_history(msgs, budget=4, count=len)                            # hàm đếm tùy chọn: đếm message
    assert len(custom) == 4 and custom[1]["role"] == "user" and n == 4
    assert len(fit_history(msgs, budget=3, count=len)[0]) == 2                    # bỏ theo cặp user/assistant


def test_truncate_middle():
    text = "Đầu văn bản. " + "nội dung ở giữa rất dài. " * 300 + "Cuối văn bản."
    out = truncate_middle(text, 60)
    assert count_tokens(out) <= 60
    assert out.startswith("Đầu văn bản.") and out.endswith("Cuối văn bản.") and " […] " in out
    assert count_tokens(truncate_middle(text, 200)) > 150                         # dùng gần hết ngân sách
    assert truncate_middle("ngắn", 10) == "ngắn"


def test_summarize_old():
    msgs = _conv(4)                                                               # system + 8 + câu cuối
    out = summarize_old(msgs, keep_last=3, summarizer=lambda old: f"{len(old)} message cũ")
    assert out[0] == msgs[0]
    assert out[1] == {"role": "system", "content": "Tóm tắt hội thoại trước: 6 message cũ"}
    assert out[2:] == msgs[-3:]
    assert summarize_old(msgs, keep_last=50, summarizer=lambda o: "x") == msgs
