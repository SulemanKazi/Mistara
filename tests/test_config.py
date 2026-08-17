"""Tuning profiles.

Every threshold in this project was fitted to one book. These tests pin the
properties that make retuning safe: partial overrides, loud rejection of typos,
real effect on behaviour, and cache invalidation when a value changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mistara.core.config import DEFAULT, MistaraConfig
from mistara.stages.s4_extract import ExtractParams
from mistara.stages.segment import assess_rows, binarize, detect_text_rows


def write(tmp_path, body: str):
    path = tmp_path / "profile.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestLoading:
    def test_none_yields_defaults(self):
        assert MistaraConfig.load(None) == DEFAULT

    def test_partial_override_keeps_every_other_default(self, tmp_path):
        cfg = MistaraConfig.load(
            write(tmp_path, 'name = "faded"\n[segment.binarize]\nblock_size = 41\n')
        )
        assert cfg.name == "faded"
        assert cfg.segment.binarize.block_size == 41
        # Untouched values must not be disturbed — a per-book profile is meant
        # to be three lines long.
        assert cfg.segment.binarize.c == DEFAULT.segment.binarize.c
        assert cfg.segment.lines.threshold_scale == DEFAULT.segment.lines.threshold_scale
        assert cfg.ingest.target_dpi == DEFAULT.ingest.target_dpi

    def test_a_misspelt_key_is_rejected_not_ignored(self, tmp_path):
        """Silently keeping the default on a typo is the worst failure mode for
        a tuning file: you would believe you had changed something."""
        with pytest.raises(ValidationError):
            MistaraConfig.load(
                write(tmp_path, "[segment.assess]\ntal_factor = 2.0\n")
            )

    def test_a_misspelt_section_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError):
            MistaraConfig.load(write(tmp_path, "[segment.assesss]\ntall_factor = 2.0\n"))

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no config at"):
            MistaraConfig.load(tmp_path / "nope.toml")

    def test_shipped_default_profile_matches_code_defaults(self):
        """`config/default.toml` documents the defaults; it must not drift."""
        from pathlib import Path

        shipped = Path("config/default.toml")
        if not shipped.exists():
            pytest.skip("default profile not present")
        loaded = MistaraConfig.load(shipped)
        assert loaded.segment == DEFAULT.segment
        assert loaded.ingest == DEFAULT.ingest
        assert loaded.extract == DEFAULT.extract

    def test_toml_round_trips(self, tmp_path):
        cfg = MistaraConfig(name="rt")
        cfg.segment.assess.tall_factor = 1.9
        path = write(tmp_path, cfg.to_toml())
        assert MistaraConfig.load(path) == cfg


class TestConfigDrivesBehaviour:
    """A knob that does not change anything is worse than no knob."""

    @staticmethod
    def page(n: int = 10) -> np.ndarray:
        import cv2

        img = np.full((40 * (n + 1), 400), 255, np.uint8)
        for i in range(n):
            cv2.rectangle(img, (20, 40 * (i + 1)), (380, 40 * (i + 1) + 20), 0, -1)
        return img

    def test_binarize_constant_changes_the_ink_mask(self):
        img = self.page()
        loose = MistaraConfig()
        loose.segment.binarize.c = 2.0
        assert (binarize(img, DEFAULT.segment) > 0).sum() != (
            binarize(img, loose.segment) > 0
        ).sum()

    def test_min_height_can_suppress_detection(self):
        img = self.page()
        strict = MistaraConfig()
        strict.segment.lines.min_height_px = 500
        assert detect_text_rows(img, DEFAULT.segment)
        assert detect_text_rows(img, strict.segment) == []

    def test_assessment_floor_is_configurable(self):
        img = self.page(n=5)
        high = MistaraConfig()
        high.segment.assess.min_rows_for_assessment = 99
        assert assess_rows(img, detect_text_rows(img), DEFAULT.segment) is not None
        assert assess_rows(img, detect_text_rows(img), high.segment) is None

    def test_pitch_threshold_reaches_the_findings(self):
        img = self.page()
        strict = MistaraConfig()
        strict.segment.assess.max_pitch_cv = -1.0  # everything is irregular now
        a = assess_rows(img, detect_text_rows(img), strict.segment)
        assert any("pitch is irregular" in f for f in a.findings())


class TestCacheInvalidation:
    def test_changing_a_threshold_changes_the_stage_params_hash(self):
        """Config is embedded in params, so a retune re-runs the stage rather
        than serving a stale cached result."""
        base = MistaraConfig()
        tweaked = MistaraConfig()
        tweaked.segment.assess.tall_factor = 1.9

        a = ExtractParams(cfg=base.extract, segment=base.segment).hash()
        b = ExtractParams(cfg=tweaked.extract, segment=tweaked.segment).hash()
        assert a != b

    def test_identical_config_hashes_identically(self):
        a = ExtractParams(cfg=DEFAULT.extract, segment=DEFAULT.segment).hash()
        b = ExtractParams(
            cfg=MistaraConfig().extract, segment=MistaraConfig().segment
        ).hash()
        assert a == b
