from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_APP = ROOT / "Scripts" / "review_app.py"


def test_review_app_compiles() -> None:
    py_compile.compile(str(REVIEW_APP), doraise=True)


def test_review_app_contains_expected_ui_labels() -> None:
    source = REVIEW_APP.read_text(encoding="utf-8")

    assert "Объединять похожие сцены" in source
    assert "Показывать все метрики" in source
    assert "Сцены" in source
    assert "Сцена" in source
    assert "sidebar_scene" in source
    assert "current_group_id" in source
