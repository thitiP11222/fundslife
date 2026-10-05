from src.final_option_strategy import FinalOptionConfig


def test_final_strategy_uses_validated_research_horizon():
    cfg = FinalOptionConfig()

    assert cfg.momentum_fast_days == 63
    assert cfg.momentum_slow_days == 126
    assert cfg.hold_days == 20


def test_final_strategy_keeps_option_selection_manual():
    cfg = FinalOptionConfig()

    assert cfg.min_dte == 30
    assert cfg.max_dte == 60
    assert cfg.top_n == 3
