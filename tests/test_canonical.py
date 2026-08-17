"""Canonicalization is load-bearing: ground truth, output, and every metric all
route through it. If it drifts, every number the project reports drifts with it,
so these are golden tests rather than smoke tests."""

from __future__ import annotations

import unicodedata

import pytest

from mistara.text.canonical import (
    Mode,
    canonical,
    repeat_rate,
    script_purity,
    strip_tashkeel,
    tashkeel_density,
    urdu_affinity,
)


def test_strict_keeps_tashkeel_loose_strips_it():
    vocalized = "بِسْمِ اللَّهِ"
    assert strip_tashkeel(vocalized) != vocalized
    assert canonical(vocalized, Mode.STRICT) != canonical(vocalized, Mode.LOOSE)
    assert "ِ" not in canonical(vocalized, Mode.LOOSE)
    assert "ِ" in canonical(vocalized, Mode.STRICT)


def test_tatweel_is_always_stripped():
    assert canonical("محـــمد", Mode.STRICT) == canonical("محمد", Mode.STRICT)


@pytest.mark.parametrize(
    ("arabic_form", "urdu_form"),
    [("كتاب", "کتاب"), ("يہ", "یہ"), ("ىہ", "یہ"), ("مسأله", "مسالہ")],
)
def test_letter_folding_unifies_arabic_and_urdu_forms(arabic_form, urdu_form):
    assert canonical(arabic_form) == canonical(urdu_form)


def test_digits_fold_to_ascii():
    assert canonical("٢٣٣") == "233"  # Arabic-Indic
    assert canonical("۲۳۳") == "233"  # Extended Arabic-Indic (Urdu)
    assert canonical("٢٣٣") == canonical("۲۳۳")


def test_output_is_nfc_normalized():
    decomposed = unicodedata.normalize("NFD", "آمین")
    assert canonical(decomposed) == unicodedata.normalize("NFC", canonical(decomposed))


def test_quranic_annotation_marks_are_removed():
    assert canonical("الٓمٓ ۝", Mode.STRICT).strip("ٓ ") != ""
    assert "۝" not in canonical("الم ۝", Mode.STRICT)


def test_whitespace_collapses_but_newlines_survive_when_asked():
    assert canonical("a   b\n\n c", keep_newlines=True) == "a b\nc"
    assert canonical("a   b\n\n c", keep_newlines=False) == "a b c"


def test_empty_input_is_safe():
    assert canonical("") == ""
    assert script_purity("") == 0.0
    assert urdu_affinity("") == 0.0


class TestScriptSignals:
    def test_urdu_affinity_separates_the_two_languages(self):
        # Urdu-only letters: ٹ ڈ ڑ ں ھ ے گ چ پ
        assert urdu_affinity("پہاڑوں کے ٹکڑے") > 0.5
        # Arabic-only forms: ة ي ك أ
        assert urdu_affinity("الكتاب المكية أية") < -0.5

    def test_tashkeel_density_flags_vocalized_quranic_text(self):
        quranic = "خُذُوا مَا آتَيْنَاكُم بِقُوَّةٍ"
        urdu_prose = "یہ معاہدہ قرآن مجید اور توراۃ دونوں میں"
        assert tashkeel_density(quranic) > tashkeel_density(urdu_prose)
        assert tashkeel_density(quranic) > 0.2

    def test_script_purity_catches_latin_and_markup(self):
        assert script_purity("یہ معاہدہ قرآن") > 0.9
        assert script_purity("Here is the translation") < 0.1
        assert script_purity('{"lines": [{"text": "x"}]}') < 0.3


class TestRepeatRate:
    def test_healthy_text_scores_low(self):
        prose = "یہ معاہدہ قرآن مجید اور توراۃ دونوں میں تصریح ہے کہ بنی اسرائیل"
        assert repeat_rate(prose) < 0.25

    def test_decoding_loop_scores_high(self):
        assert repeat_rate("نمونہ متن " * 40) > 0.5

    def test_short_text_is_not_penalized(self):
        assert repeat_rate("مختصر") == 0.0
