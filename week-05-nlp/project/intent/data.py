"""Dữ liệu tổng hợp: tin nhắn khách hàng gửi tổng đài nhà mạng (tiếng Việt), gán intent.

Có sẵn, KHÔNG cần sửa. Mỗi tin nhắn sinh ra từ một template (câu mẫu) + điền slot + nhiễu:
bỏ dấu, teencode, gõ sai, Unicode NFD, hai kiểu đặt dấu thanh ("hủy" / "huỷ"), lời chào, emoji.
Một số template nhập nhằng: nhãn được rút ngẫu nhiên theo xác suất, giống như hai người gán nhãn không đồng ý.

    >>> df = make_messages(6000, seed=0)
    >>> df.columns.tolist()
    ['text', 'intent', 'template_id', 'nodiac', 'teen', 'nfd']

Tên gói cước, số điện thoại, email đều là giả định.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

INTENTS = ["tra_cuoc", "nap_tien", "dang_ky_goi", "huy_goi", "loi_mang", "doi_sim", "khieu_nai", "gap_tvv", "khac"]
IN_SCOPE = INTENTS[:-1]                     # "khac" = ngoài phạm vi (out-of-scope)

# (câu mẫu, nhãn) hoặc (câu mẫu, {nhãn: xác suất}) cho câu nhập nhằng
TEMPLATES: dict[str, list] = {
    "tra_cuoc": [
        "kiểm tra số dư tài khoản", "tài khoản của tôi còn bao nhiêu tiền", "xem giúp tôi cước tháng {thang}",
        "cước tháng này của số {sdt} là bao nhiêu", "tôi muốn tra cứu cước trả sau",
        "còn bao nhiêu data trong gói {goi}", "dung lượng data còn lại bao nhiêu", "gói {goi} của tôi còn hạn đến ngày nào",
        "làm sao để xem số dư", "tháng này tôi đã dùng hết bao nhiêu tiền", "check tài khoản chính giúp mình",
        "số tiền khuyến mãi còn bao nhiêu", "hóa đơn tháng {thang} bao nhiêu tiền", "tra số dư bằng cú pháp nào",
        "mình đang dùng gói gì vậy", "kiểm tra lưu lượng còn lại của sim",
    ],
    "nap_tien": [
        "nạp {tien} vào tài khoản", "tôi muốn nạp tiền điện thoại", "nạp thẻ cào thế nào", "nạp tiền cho số {sdt}",
        "mua thẻ {tien} ở đâu", "có nạp tiền qua ngân hàng được không", "nạp hộ tôi {tien} cho thuê bao {sdt}",
        "cú pháp nạp thẻ là gì", "thanh toán cước trả sau qua ví điện tử", "nạp tiền bằng thẻ visa được không",
        "tôi cần nạp thêm tiền để gọi", "muốn mua mã thẻ online", "hướng dẫn nạp tiền qua app",
        "chuyển tiền nạp cho người thân", "nạp tiền có được khuyến mãi không",
    ],
    "dang_ky_goi": [
        "đăng ký gói {goi}", "tôi muốn đăng ký gói data", "mua thêm data", "cho tôi gói cước rẻ nhất",
        "gói nào nhiều data nhất", "đăng ký gói {goi} cho số {sdt}", "có gói nào gọi nội mạng miễn phí không",
        "muốn mua gói {goi} theo tháng", "gia hạn gói {goi}", "đăng ký lại gói {goi}",
        ("hết data rồi muốn mua thêm", {"dang_ky_goi": 0.85, "loi_mang": 0.15}), "nâng cấp lên gói {goi}",
        "tư vấn gói cước cho sinh viên", "cú pháp đăng ký gói {goi}", "kích hoạt gói {goi} giúp mình",
        "gói 5G đăng ký thế nào",
    ],
    "huy_goi": [
        "hủy gói {goi}", "tôi muốn hủy gói data", "không muốn dùng gói {goi} nữa", "dừng gia hạn gói {goi}",
        "tắt tự động gia hạn", "hủy dịch vụ nhạc chờ", "cắt gói {goi} giúp tôi", "làm sao để hủy gói cước",
        "ngừng sử dụng gói {goi} từ tháng sau", "hủy các dịch vụ giá trị gia tăng",
        ("tôi không đăng ký gói này sao lại bị gia hạn, hủy giúp tôi", {"huy_goi": 0.6, "khieu_nai": 0.4}),
        "bỏ gói {goi} đi", "cú pháp hủy gói {goi}", "hủy đăng ký gói {goi}", "không gia hạn gói {goi} nữa",
        "chuyển từ gói {goi} về gói thường",
    ],
    "loi_mang": [
        "mạng yếu quá", "không vào được mạng {mang}", "mất sóng {noi}", "{tg} không gọi được",
        "điện thoại báo không có dịch vụ", "mạng chập chờn {tg}", "vào mạng rất chậm", "không nhận được tin nhắn otp",
        "gọi điện bị rớt cuộc liên tục", "sóng {mang} {noi} rất kém", "không bắt được {mang}",
        "nghe không rõ, tiếng bị rè",
        ("data vẫn còn mà không vào mạng được", {"loi_mang": 0.8, "tra_cuoc": 0.2}),
        ("sim báo lỗi không có tín hiệu", {"loi_mang": 0.6, "doi_sim": 0.4}),
        "mạng {mang} bị lỗi à", "{tg} mạng cứ mất liên tục",
    ],
    "doi_sim": [
        "đổi sim", "tôi bị mất sim", "muốn chuyển sang esim", "làm lại sim ở đâu", "sim bị hỏng cần đổi sim mới",
        "đổi sim 4G", "đổi sim cần mang giấy tờ gì", "chuyển từ sim thường sang esim", "sim bị khóa vì nhập sai mã puk",
        "đổi sim online được không", "cắt sim lại cho vừa máy", "mất điện thoại muốn khóa sim",
        "cấp lại sim cho số {sdt}", "đăng ký thông tin thuê bao chính chủ", "đổi sim mất phí bao nhiêu",
    ],
    "khieu_nai": [
        "sao tài khoản tôi bị trừ tiền", "bị trừ {tien} vô lý", "tôi không đăng ký mà vẫn bị trừ tiền",
        "khiếu nại cước tháng {thang}", "cước tháng này cao bất thường",
        ("nạp {tien} rồi mà chưa thấy cộng tiền", {"khieu_nai": 0.75, "nap_tien": 0.25}),
        "tự nhiên mất hết tiền trong tài khoản", "tôi muốn khiếu nại về chất lượng dịch vụ", "bị tính cước sai",
        "sao gói {goi} trừ tiền hai lần", "đòi hoàn tiền cước bị trừ sai", "dịch vụ quá tệ, tôi muốn phản ánh",
        "trừ tiền mà không có tin nhắn thông báo",
        ("tại sao hóa đơn tháng {thang} lại tăng", {"khieu_nai": 0.6, "tra_cuoc": 0.4}),
        "nhân viên tư vấn thái độ không tốt", "gửi phản ánh qua email {email} được không",
    ],
    "gap_tvv": [
        "cho tôi gặp tổng đài viên", "gặp nhân viên", "tôi muốn nói chuyện với người thật",
        "chuyển máy cho nhân viên hỗ trợ", "không muốn nói chuyện với máy", "kết nối tư vấn viên giúp tôi",
        "có ai hỗ trợ trực tiếp không", "gọi lại cho tôi số {sdt}", "tôi cần nhân viên gọi lại",
        "bot trả lời không đúng, cho gặp người", "gặp chăm sóc khách hàng", "nối máy tổng đài viên",
        "nhờ nhân viên xử lý trực tiếp", "số hotline chăm sóc khách hàng là gì",
    ],
    "khac": [
        "hôm nay trời có mưa không", "kể chuyện cười đi", "đặt vé máy bay đi Đà Nẵng", "cách nấu phở bò",
        "bạn tên là gì", "giá vàng hôm nay", "tỷ số trận bóng tối qua", "dịch câu này sang tiếng anh",
        "mua điện thoại trả góp ở đâu", "quán cà phê nào gần đây", "bạn có người yêu chưa",
        "viết giúp tôi một bài thơ", "lịch chiếu phim cuối tuần", "đường đến bệnh viện gần nhất",
        "tính giúp tôi 15 nhân 7", "mình buồn quá", "mở giúp trang {url}",
    ],
}

SLOTS = {
    "goi": ["GX30", "GX70", "TOP90", "MINI10", "PLUS120", "SV50", "5G150"],
    "tien": ["50k", "100k", "200k", "500k", "20 nghìn", "100 nghìn", "200.000đ", "50.000 đồng"],
    "sdt": ["0912345678", "0987654321", "0901234567", "0363333444", "0868 123 456", "+84912000111"],
    "thang": ["này", "trước", "3", "10", "12"],
    "mang": ["4G", "5G", "3G", "internet", "data"],
    "noi": ["ở nhà", "ở quận 7", "trong phòng", "ở chung cư", "ở quê", "trên xe buýt"],
    "tg": ["từ sáng", "mấy hôm nay", "từ hôm qua", "tối nào cũng", "cả tuần nay"],
    "email": ["an.nguyen@example.com", "hotro@example.vn"],
    "url": ["https://example.com/khuyen-mai", "www.example.vn"],
}
SYNONYMS = {"tôi": ["tôi", "mình", "em", "tôi"], "muốn": ["muốn", "cần", "muốn"], "giúp": ["giúp", "hộ", "giùm"]}
PREFIX = ["", "", "", "alo ", "ad ơi ", "cho mình hỏi ", "em ơi ", "chào tổng đài, ", "xin hỏi ", "tổng đài ơi "]
SUFFIX = ["", "", "", " ạ", " nhé", " với", " giúp mình", " được không", "?", " ??", "!!", " 🙏", " gấp", " 😡"]
# viết tắt kiểu chat ("teencode"): từ chuẩn → các cách viết
TEEN = {"không": ["ko", "k", "hông", "hok"], "được": ["dc", "đc", "đk"], "gì": ["j"], "rồi": ["r", "rùi"],
        "vậy": ["v"], "mình": ["mk", "mik"], "tôi": ["t", "tui"], "bao nhiêu": ["bn", "bnh"], "với": ["vs"],
        "nhé": ["nha", "nhe"], "thế nào": ["ntn"], "nhân viên": ["nv"], "tài khoản": ["tk"], "điện thoại": ["đt"],
        "biết": ["bít"], "giúp": ["júp"], "tiền": ["tiền", "xiền"]}
# hai kiểu đặt dấu thanh cho cụm "uy"/"oa"/"oe": cũ (dấu trên âm đầu) ↔ mới (dấu trên âm sau)
TONE_STYLE = {"ủy": "uỷ", "úy": "uý", "ùy": "uỳ", "ũy": "uỹ", "ụy": "uỵ", "òa": "oà", "óa": "oá", "ỏa": "oả",
              "õa": "oã", "ọa": "oạ", "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ"}


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = int(rng.integers(1, len(word) - 1))
    if rng.random() < 0.5:                                   # xóa một ký tự
        return word[:i] + word[i + 1:]
    return word[:i - 1] + word[i] + word[i - 1] + word[i + 1:]   # đảo hai ký tự kề nhau


def _render(template: str, rng: np.random.Generator) -> tuple[str, bool, bool, bool]:
    text = template
    for slot, values in SLOTS.items():
        while "{" + slot + "}" in text:
            text = text.replace("{" + slot + "}", values[int(rng.integers(len(values)))], 1)
    for word, alts in SYNONYMS.items():
        if f" {word} " in f" {text} " and rng.random() < 0.5:
            text = f" {text} ".replace(f" {word} ", f" {alts[int(rng.integers(len(alts)))]} ", 1).strip()
    text = PREFIX[int(rng.integers(len(PREFIX)))] + text + SUFFIX[int(rng.integers(len(SUFFIX)))]
    teen = rng.random() < 0.35
    if teen:
        for word, alts in TEEN.items():
            if word in text and rng.random() < 0.7:
                text = text.replace(word, alts[int(rng.integers(len(alts)))])
    if rng.random() < 0.3:
        for old, new in TONE_STYLE.items():
            text = text.replace(old, new)
    if rng.random() < 0.15:
        words = text.split(" ")
        j = int(rng.integers(len(words)))
        words[j] = _typo(words[j], rng)
        text = " ".join(words)
    nodiac = rng.random() < 0.3
    if nodiac:
        text = _strip_accents(text)
    r = rng.random()
    if r < 0.10:
        text = text[:1].upper() + text[1:]
    elif r < 0.13:
        text = text.upper()
    if rng.random() < 0.08:
        text = text.replace(" ", "  ", 1)
    nfd = rng.random() < 0.10
    if nfd:
        text = unicodedata.normalize("NFD", text)
    return text, nodiac, teen, nfd


def make_messages(n: int = 6000, seed: int = 0, ood_frac: float = 0.10) -> pd.DataFrame:
    """Sinh `n` tin nhắn. Intent trong phạm vi chia đều; `ood_frac` là tỉ lệ tin nhắn "khac" (ngoài phạm vi)."""
    rng = np.random.default_rng(seed)
    p_int = np.r_[np.full(len(IN_SCOPE), (1 - ood_frac) / len(IN_SCOPE)), ood_frac]
    rows = []
    for _ in range(n):
        name = INTENTS[int(rng.choice(len(INTENTS), p=p_int))]
        k = int(rng.integers(len(TEMPLATES[name])))
        item = TEMPLATES[name][k]
        template, labels = (item, {name: 1.0}) if isinstance(item, str) else item
        names = list(labels)
        label = names[int(rng.choice(len(names), p=list(labels.values())))]
        text, nodiac, teen, nfd = _render(template, rng)
        rows.append((text, label, f"{name}/{k:02d}", nodiac, teen, nfd))
    return pd.DataFrame(rows, columns=["text", "intent", "template_id", "nodiac", "teen", "nfd"])
