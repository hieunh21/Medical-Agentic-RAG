"""RateLimiter của API công khai: giới hạn theo IP mỗi giờ + trần cả hệ thống mỗi ngày (UTC)."""
from medical_agentic_rag.api.app import RateLimiter

DAY = 86_400
T0 = 1_800_000_000.0  # một thời điểm cố định, giữa ngày UTC (08:00)


def test_tat_khi_bang_0():
    lim = RateLimiter(per_ip_hour=0, per_day=0)
    assert all(lim.check("1.1.1.1", T0 + i) is None for i in range(500))


def test_moi_ip_toi_da_n_cau_moi_gio():
    lim = RateLimiter(per_ip_hour=3, per_day=0)
    assert [lim.check("a", T0 + i) for i in range(3)] == [None] * 3
    assert lim.check("a", T0 + 10) is not None
    assert lim.check("b", T0 + 10) is None  # IP khác không bị ảnh hưởng


def test_cua_so_mot_gio_truot():
    lim = RateLimiter(per_ip_hour=2, per_day=0)
    lim.check("a", T0)
    lim.check("a", T0 + 1800)
    assert lim.check("a", T0 + 3599) is not None
    assert lim.check("a", T0 + 3601) is None  # lượt đầu đã ra khỏi cửa sổ


def test_tran_ngay_cho_ca_he_thong():
    lim = RateLimiter(per_ip_hour=0, per_day=2)
    assert lim.check("a", T0) is None
    assert lim.check("b", T0 + 1) is None
    assert "hôm nay" in lim.check("c", T0 + 2)


def test_sang_ngay_moi_thi_dem_lai():
    lim = RateLimiter(per_ip_hour=1, per_day=1)
    lim.check("a", T0)
    assert lim.check("a", T0 + 60) is not None
    assert lim.check("a", T0 + DAY) is None


def test_lan_bi_tu_choi_khong_bi_tinh_vao_tran_ngay():
    lim = RateLimiter(per_ip_hour=1, per_day=3)
    lim.check("a", T0)
    for i in range(5):  # IP "a" bị chặn theo giờ: không được ăn mất lượt của người khác
        lim.check("a", T0 + 1 + i)
    assert lim.check("b", T0 + 10) is None
    assert lim.check("c", T0 + 11) is None
