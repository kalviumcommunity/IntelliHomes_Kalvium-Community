import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_experiment_cases_contains_expected_parameters():
    prompt_test = importlib.import_module("prompt_test")

    cases = prompt_test.build_experiment_cases()

    assert [case["name"] for case in cases] == [
        "temperature_low",
        "temperature_high",
        "max_tokens_short",
        "stop_truncated",
    ]

    assert cases[0]["parameters"]["temperature"] == 0.0
    assert cases[2]["parameters"]["max_tokens"] == 40
    assert cases[3]["parameters"]["stop"] == ["\n\n"]


def test_format_experiment_report_includes_effect_summary():
    prompt_test = importlib.import_module("prompt_test")

    case = {
        "name": "temperature_low",
        "parameters": {"temperature": 0.0},
        "output": "1. Title deed\n2. Sale agreement",
        "effect": "Stable and factual",
    }

    report = prompt_test.format_experiment_report([case])

    assert "temperature_low" in report
    assert "Stable and factual" in report
