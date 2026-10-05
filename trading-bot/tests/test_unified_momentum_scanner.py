from src.unified_momentum_scanner import UnifiedMomentumConfig


def test_unified_scanner_keeps_validated_core_momentum():
    cfg = UnifiedMomentumConfig()

    assert cfg.momentum_fast_days == 63
    assert cfg.momentum_slow_days == 126
    assert cfg.momentum_fast_weight == 0.60
    assert cfg.momentum_slow_weight == 0.40


def test_options_are_more_selective_than_fractional_core():
    cfg = UnifiedMomentumConfig()

    assert cfg.top_n == 3
    assert cfg.near_52w_high_ratio == 0.90
    assert cfg.min_price_usd >= 5.0
