"""S6a — normalization, alignment, and the replacement policy.

Most tests build a tiny synthetic corpus so the right answer is known by
construction. The handful that need the real Quran are guarded: the corpus is
fetched, not vendored, so it may legitimately be absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mistara.core.config import MatchConfig
from mistara.core.model import BBox, Language, Line, ProvenanceKind, Span
from mistara.corpus.align import identity, smith_waterman
from mistara.corpus.index import CorpusIndex
from mistara.corpus.quran import CorpusMatcher
from mistara.stages.s6a_match import CITED, REPLACED, LineMatch, _rewrite, decide
from mistara.text.arabic import match_form, match_tokens, tokenize

CORPUS_PATH = Path("data/corpora/quran.json")
needs_corpus = pytest.mark.skipif(
    not CORPUS_PATH.is_file(), reason="run `mistara corpus fetch` first"
)

#: Al-Ikhlas plus one Al-Baqara verse — enough to exercise seeding and ambiguity.
VERSES = [
    (112, 1, "قُلۡ هُوَ ٱللَّهُ أَحَدٌ"),
    (112, 2, "ٱللَّهُ ٱلصَّمَدُ"),
    (112, 3, "لَمۡ يَلِدۡ وَلَمۡ يُولَدۡ"),
    (112, 4, "وَلَمۡ يَكُن لَّهُۥ كُفُوًا أَحَدُۢ"),
    (2, 255, "ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلۡحَيُّ ٱلۡقَيُّومُ"),
]


@pytest.fixture(scope="module")
def small_index() -> CorpusIndex:
    return CorpusIndex.from_verses("quran", VERSES)


# --------------------------------------------------------------------------- #
# Normalization — the part that must be identical on both sides
# --------------------------------------------------------------------------- #


def test_uthmani_and_urdu_typeface_normalize_identically():
    """The whole stage rests on this: two editions of one verse must agree.

    Left is the corpus (Uthmani: wasla alef, dagger alef, Arabic yeh/kaf/heh);
    right is how the same words come off an Urdu-typeset page.
    """
    uthmani = "بِـَٔايَٰتِي ثَمَنٗا قَلِيلٗاۚ وَمَن لَّمۡ يَحۡكُم"
    urdu_set = "بِاٰیٰتِیْ ثَمَنًا قَلِیْلًا وَمَنْ لَّمْ یَحْکُمْ"
    assert match_form(uthmani) == match_form(urdu_set)


def test_normalization_drops_every_combining_mark():
    # Note ٱ -> ا and ه -> ہ: the fold targets are arbitrary, and only agreement
    # between corpus and query matters.
    assert match_form("قُلۡ هُوَ ٱللَّهُ أَحَدٌ") == "قل ہو اللہ احد"


def test_normalization_drops_digits_ornaments_and_punctuation():
    assert match_form("﴿الْکٰفِرُوْنَ﴾ (5 مائدہ 44)") == "الکفرون مایدہ"


def test_dagger_alef_is_stripped_on_both_sides_not_expanded():
    """`کٰفِرُوْنَ` loses its dagger alef rather than gaining a full one.

    That looks lossy in isolation and is exactly right in context: the Uthmani
    corpus spells the same word `ٱلۡكَٰفِرُونَ`, with the same dagger alef, so both
    sides land on the same consonantal skeleton. Normalization only has to be
    *consistent*, never faithful.
    """
    assert match_form("الْکٰفِرُوْنَ") == match_form("ٱلۡكَٰفِرُونَ") == "الکفرون"


def test_tokenize_keeps_offsets_into_the_original_string():
    text = "قال: وَلَا تَشْتَرُوْا هنا"
    tokens = tokenize(text)
    for token in tokens:
        assert match_form(text[token.start : token.end]).replace(" ", "") == token.text


def test_empty_and_non_arabic_text_yields_no_tokens():
    assert match_tokens("") == []
    assert match_tokens("hello world 123") == []


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def test_smith_waterman_finds_a_substring():
    target = ["a", "b", "c", "d", "e", "f"]
    aln = smith_waterman(["c", "d", "e"], target)
    assert (aln.t_start, aln.t_end) == (2, 5)
    assert (aln.q_start, aln.q_end) == (0, 3)


def test_smith_waterman_is_local_on_both_sequences():
    """An inline quote: the match sits inside the query as well as the target."""
    aln = smith_waterman(["xx", "yy", "c", "d", "e", "zz"], ["a", "b", "c", "d", "e", "f"])
    assert (aln.q_start, aln.q_end) == (2, 5)
    assert (aln.t_start, aln.t_end) == (2, 5)


def test_smith_waterman_tolerates_one_bad_token():
    """One misread word must not throw away an otherwise perfect verse."""
    aln = smith_waterman(["قل", "ہو", "اللہ", "احد"], ["قل", "ہو", "الله", "احد"])
    assert aln.q_end - aln.q_start == 4


def test_smith_waterman_returns_nothing_on_unrelated_input():
    assert smith_waterman(["aaa", "bbb", "ccc"], ["xxx", "yyy", "zzz"]) is None


def test_identity_is_character_level():
    assert identity(["abcd"], ["abcd"]) == 1.0
    assert 0.0 < identity(["abcd"], ["abce"]) < 1.0


# --------------------------------------------------------------------------- #
# Index and matcher
# --------------------------------------------------------------------------- #


def test_index_streams_tokens_continuously_across_verses(small_index):
    assert len(small_index) == sum(len(match_tokens(v[2])) for v in VERSES)
    assert small_index.ref_for(0, 4) == "quran:112:1"


def test_reference_spans_verses_when_a_quote_crosses_a_boundary(small_index):
    # Last token of 112:1 through the first of 112:2.
    start = len(match_tokens(VERSES[0][2])) - 1
    assert small_index.ref_for(start, start + 2) == "quran:112:1-2"


def test_matcher_recovers_a_verse_read_in_an_urdu_typeface(small_index):
    m = CorpusMatcher(small_index)
    result = m.match("لَمْ یَلِدْ وَلَمْ یُوْلَدْ")
    assert result.ref == "quran:112:3"
    assert result.identity == pytest.approx(1.0)
    assert result.coverage == pytest.approx(1.0)


def test_matcher_returns_verbatim_corpus_text_with_diacritics(small_index):
    result = CorpusMatcher(small_index).match("لَمْ یَلِدْ وَلَمْ یُوْلَدْ")
    assert result.corpus_text == "لَمۡ يَلِدۡ وَلَمۡ يُولَدۡ"


def test_matcher_locates_an_inline_quote_inside_prose(small_index):
    line = "اس آیت میں لَمْ یَلِدْ وَلَمْ یُوْلَدْ کا مفہوم یہ ہے"
    result = CorpusMatcher(small_index).match(line)
    assert line[result.char_start : result.char_end] == "لَمْ یَلِدْ وَلَمْ یُوْلَدْ"
    assert result.coverage < 0.85  # a quote *inside* a line, not the whole line


def test_matcher_declines_queries_below_the_token_floor(small_index):
    assert CorpusMatcher(small_index, min_query_tokens=3).match("قل ہو") is None


def test_a_two_word_tail_matches_only_as_a_continuation(small_index):
    """The 1b safety property, end to end. Two words of a verse are below the
    cold lookup floor and return nothing on their own — which is exactly what
    stops an isolated short phrase being matched. But immediately after the line
    they continue, the contiguity context licenses the lookup and finds them."""
    m = CorpusMatcher(small_index)  # default min_query_tokens=3
    tail = "ٱلۡحَيُّ ٱلۡقَيُّومُ"  # the last two words of 2:255
    assert m.match(tail) is None  # cold: below the query floor, so untouchable

    head = m.match("ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ")  # anchor into 2:255
    cont = m.match(tail, prefer_after=head.corpus_end)
    assert cont is not None
    assert cont.ref == "quran:2:255"
    assert 0 <= cont.corpus_start - head.corpus_end <= 3  # genuinely contiguous


def test_candidates_seed_short_queries_from_rare_unigrams(small_index):
    # A one- or two-word tail has no trigram; the rare-unigram fallback is what
    # lets a continuation be found at all. An empty query still yields nothing.
    assert small_index.candidates(match_tokens("ٱلۡقَيُّومُ"))
    assert small_index.candidates([]) == []


# --------------------------------------------------------------------------- #
# Policy — where the false-replacement guarantee lives
# --------------------------------------------------------------------------- #


def _match(**kw):
    from mistara.corpus.quran import Match

    base = dict(
        ref="quran:2:1", identity=1.0, coverage=1.0, margin=1.0, candidates=1,
        score=5.0, char_start=0, char_end=10, corpus_start=0, corpus_end=5,
        corpus_text="x", tokens_matched=5, query_tokens=5,
    )
    return Match(**{**base, **kw})


def test_a_whole_line_verse_is_replaced():
    assert decide(_match(), MatchConfig()) == REPLACED


def test_an_inline_quote_is_cited_not_replaced_by_default():
    assert decide(_match(coverage=0.3), MatchConfig()) == CITED


def test_inline_replacement_is_available_but_opt_in():
    cfg = MatchConfig(replace_inline=True)
    assert decide(_match(coverage=0.3), cfg) == REPLACED


def test_low_identity_is_never_acted_on():
    assert decide(_match(identity=0.5), MatchConfig()) is None


def test_a_short_span_is_rejected_however_perfect():
    """The measured hazard: one common word matches Urdu prose at identity 1.0.

    Coverage cannot catch this, because a real inline quote is also a
    low-coverage match. Only span length separates them.
    """
    assert decide(_match(tokens_matched=1, coverage=0.08), MatchConfig()) is None


def test_a_short_span_is_acted_on_when_it_continues_a_passage():
    """1b: a short line is safe to act on when it contiguously continues the
    passage the previous accepted line matched — continuity is the evidence the
    length floor gives up, and a random short line will not also land on the
    exact next corpus position."""
    cfg = MatchConfig()
    assert decide(_match(tokens_matched=1, coverage=1.0), cfg, continues=True) == REPLACED
    # Same match, not a continuation: the isolated-line floor still refuses it.
    assert decide(_match(tokens_matched=1, coverage=1.0), cfg, continues=False) is None


def test_a_continuation_is_replaced_not_merely_cited_at_low_coverage():
    """A verse tail like `مِمَّا یَکْسِبُوْنَ (79 بقرہ)` has low coverage — the
    citation is unmatched query — but continuity vouches for its location and the
    aligner bounds the verse span exactly, so it is restored (green), not just
    cited. An isolated low-coverage match is still only cited."""
    cfg = MatchConfig()
    assert decide(_match(tokens_matched=2, coverage=0.5), cfg, continues=True) == REPLACED
    assert decide(_match(tokens_matched=5, coverage=0.5), cfg, continues=False) == CITED


def test_continuity_vouches_for_the_location_never_for_a_bad_reading():
    assert decide(_match(identity=0.5, tokens_matched=1), MatchConfig(), continues=True) is None


def test_a_low_margin_does_not_block_replacement():
    """Ambiguity is about the citation, not the wording.

    `وَمَنْ لَّمْ یَحْکُمْ بِمَآ اَنْزَلَ اللّٰہُ` is identical at 5:44, 5:45 and 5:47 —
    the text is certain even though the reference is not.
    """
    assert decide(_match(margin=0.0), MatchConfig()) == REPLACED


# --------------------------------------------------------------------------- #
# Writeback
# --------------------------------------------------------------------------- #


def _line(text: str) -> Line:
    return Line(
        line_id="L0001",
        bbox=BBox(x=0, y=0, w=100, h=10),
        text=text,
        spans=[Span(span_id="L0001s0", char_start=0, char_end=len(text))],
    )


def _line_match(**kw) -> LineMatch:
    base = dict(
        page_no=0, region_id="r0", line_id="L0001", ref="quran:112:3",
        identity=1.0, coverage=1.0, margin=1.0, tokens_matched=4,
        action=REPLACED, corpus_start=0, corpus_end=4,
        char_start=0, char_end=0, corpus_text="",
    )
    return LineMatch(**{**base, **kw})


def test_replacement_keeps_text_outside_the_matched_span():
    """`الکافرون (5 مائدہ 44)` must keep the book's own citation beside the verse."""
    line = _line("قل ہو اللہ احد (112 اخلاص 1)")
    match = _line_match(char_start=0, char_end=14, corpus_text="قُلۡ هُوَ ٱللَّهُ أَحَدٌ")
    out = _rewrite(line, match)
    assert out.text == "قُلۡ هُوَ ٱللَّهُ أَحَدٌ (112 اخلاص 1)"
    assert out.spans[0].provenance.kind is ProvenanceKind.QURAN
    assert out.spans[1].provenance.kind is ProvenanceKind.OCR


def test_replacement_retains_the_ocr_reading():
    line = _line("قل ہو اللہ احد")
    out = _rewrite(line, _line_match(char_end=14, corpus_text="قُلۡ هُوَ ٱللَّهُ أَحَدٌ"))
    assert out.spans[0].provenance.original_text == "قل ہو اللہ احد"


def test_citation_leaves_the_text_alone():
    line = _line("اس میں قل ہو اللہ احد ہے")
    out = _rewrite(line, _line_match(action=CITED, char_start=7, char_end=21))
    assert out.text == line.text
    sacred = next(s for s in out.spans if s.provenance.source_ref)
    assert sacred.provenance.kind is ProvenanceKind.OCR  # text is still OCR
    assert sacred.provenance.source_ref == "quran:112:3"


def test_rerunning_does_not_destroy_the_original_reading():
    """Provenance is the audit trail; a second pass must not overwrite it."""
    line = _line("قل ہو اللہ احد")
    match = _line_match(char_end=14, corpus_text="قُلۡ هُوَ ٱللَّهُ أَحَدٌ")
    once = _rewrite(line, match)
    twice = _rewrite(once, _line_match(char_end=len(once.text), corpus_text=match.corpus_text))
    assert twice.spans[0].provenance.original_text == "قل ہو اللہ احد"


def test_a_span_outside_the_line_is_ignored_rather_than_crashing():
    line = _line("قل ہو")
    assert _rewrite(line, _line_match(char_start=0, char_end=999)) is line


def test_surrounding_prose_keeps_its_language_tag():
    """An inline quote must not strip the language off the prose either side of
    it. Before this, `_rewrite` reset the surrounding spans to UNKNOWN, so a
    mixed line's `match` output was nonsense even before S5 re-routed it."""
    line = Line(
        line_id="L0001",
        bbox=BBox(x=0, y=0, w=100, h=10),
        text="اس میں قل ہو اللہ احد ہے",
        spans=[
            Span(
                span_id="L0001s0",
                char_start=0,
                char_end=24,
                language=Language.URDU,
            )
        ],
    )
    out = _rewrite(line, _line_match(action=CITED, char_start=7, char_end=21))
    non_sacred = [s for s in out.spans if not s.provenance.source_ref]
    assert non_sacred and all(s.language is Language.URDU for s in non_sacred)
    sacred = next(s for s in out.spans if s.provenance.source_ref)
    assert sacred.language is Language.ARABIC


# --------------------------------------------------------------------------- #
# Against the real corpus
# --------------------------------------------------------------------------- #


@needs_corpus
def test_real_corpus_is_complete():
    from mistara.corpus.quran import load_quran

    index = load_quran()
    assert len(index.refs) == 6236
    assert len(index) > 70_000


@needs_corpus
def test_repeated_quranic_wording_reports_an_honest_margin():
    """5:44, 5:45 and 5:47 share this clause verbatim — the citation is a guess."""
    from mistara.corpus.quran import load_quran

    result = CorpusMatcher(load_quran()).match(
        "وَمَنْ لَّمْ یَحْکُمْ بِمَآ اَنْزَلَ اللّٰہُ"
    )
    assert result.identity == pytest.approx(1.0)
    assert result.margin < 0.05


@needs_corpus
def test_urdu_prose_is_not_replaced():
    """The matcher runs on every line; policy is what keeps prose safe."""
    from mistara.corpus.quran import load_quran

    matcher = CorpusMatcher(load_quran())
    cfg = MatchConfig()
    for prose in (
        "یہ یہود کے نقشِ عبدیت کی دوسری مثال بیان ہو رہی ہے",
        "اس آیت میں مطلب یہ نہیں ہے کہ اگر سود در سود کی شکل پیدا نہ ہو",
        "اہل تاویل کے درمیان اس امر میں اختلاف ہوا ہے کہ یہ مسخ ہو گیا تھا",
    ):
        result = matcher.match(prose)
        assert result is None or decide(result, cfg) is None
