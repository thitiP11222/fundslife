from src.risk import position_size_from_risk


def test_position_size_respects_stop_risk():
    shares = position_size_from_risk(
        equity=10_000,
        risk_fraction=0.01,
        entry_price=100,
        stop_price=98,
        max_position_value_pct=1.0,
    )
    assert shares == 50


def test_position_size_respects_notional_cap():
    shares = position_size_from_risk(
        equity=10_000,
        risk_fraction=0.10,
        entry_price=200,
        stop_price=190,
        max_position_value_pct=0.20,
    )
    assert shares == 10


def test_fractional_position_size_for_small_account():
    shares = position_size_from_risk(
        equity=30,
        risk_fraction=0.005,
        entry_price=240,
        stop_price=235,
        max_position_value_pct=0.90,
        allow_fractional=True,
        share_step=0.001,
    )
    assert 0 < shares < 1
    assert round(shares, 3) == shares
