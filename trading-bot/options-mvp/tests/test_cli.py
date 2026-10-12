from scan_cli import run_scan

def test_empty_scan():
    assert run_scan([])==[]
