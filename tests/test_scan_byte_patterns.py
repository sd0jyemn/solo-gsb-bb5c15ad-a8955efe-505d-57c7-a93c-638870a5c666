"""Tests for byte-pattern compilation/scanning and /api/v1/scan-byte-patterns."""

from fastapi.testclient import TestClient

from app.byte_scan import (
    ADDR_SPAN,
    MAX_PATTERN_BYTES,
    compile_pattern,
    find_hits,
    validate_scope,
)
from app.hex_parser import Segment, parse_intel_hex
from app.main import app

client = TestClient(app)


def rec(rectype: int, addr: int, data: bytes = b"") -> str:
    """Build a valid Intel HEX record line."""
    body = bytes([len(data), (addr >> 8) & 0xFF, addr & 0xFF, rectype]) + bytes(data)
    checksum = (-sum(body)) & 0xFF
    return ":" + (body + bytes([checksum])).hex().upper()


EOF = rec(0x01, 0)


def data_rec(addr: int, data: bytes) -> str:
    return rec(0x00, addr, data)


def linear_base(value: int) -> str:
    return rec(0x04, 0, value.to_bytes(2, "big"))


SINGLE_SEGMENT_HEX = (
    ":020000040800F2\n"
    ":10000000214601360121470136007EFE09D2190141\n"
    ":00000001FF\n"
)
SEGMENT_START = 0x08000000


def seg(start: int, data: bytes) -> Segment:
    return Segment(start=start, data=data)


# ---------------------------------------------------------------------------
# pattern compilation
# ---------------------------------------------------------------------------

def test_compile_plain_hex_bytes():
    p = compile_pattern("p", "AB cd 00 FF")
    assert p.valid
    assert p.length == 4
    assert p.values == bytes([0xAB, 0xCD, 0x00, 0xFF])
    assert p.masks == b"\xff\xff\xff\xff"
    assert p.errors == []


def test_compile_wildcards():
    p = compile_pattern("p", "?? AB ??")
    assert p.valid
    assert p.length == 3
    assert p.values == bytes([0x00, 0xAB, 0x00])
    assert p.masks == bytes([0x00, 0xFF, 0x00])


def test_compile_repeats_for_fixed_and_wildcard_tokens():
    p = compile_pattern("p", "FF{3} ??{2} AB")
    assert p.valid
    assert p.length == 6
    assert p.values == bytes([0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xAB])
    assert p.masks == bytes([0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF])


def test_compile_accepts_extra_and_mixed_whitespace_and_no_whitespace():
    assert compile_pattern("p", "  12\t34\n").values == bytes([0x12, 0x34])
    p = compile_pattern("p", "12{2}??{2}")
    assert p.valid
    assert p.values == bytes([0x12, 0x12, 0x00, 0x00])
    assert p.masks == bytes([0xFF, 0xFF, 0x00, 0x00])


def test_compile_empty_and_whitespace_only_patterns_are_invalid():
    for syntax in ("", "   ", "\t\n"):
        p = compile_pattern("p", syntax)
        assert not p.valid
        assert p.length == 0
        assert p.values == b""
        assert [e.kind for e in p.errors] == ["invalid_pattern_syntax"]


def test_compile_collects_multiple_syntax_diagnostics_without_scanning():
    p = compile_pattern("p", "AB ??{0} {2} x ??{")
    kinds = [e.kind for e in p.errors]
    assert kinds == [
        "invalid_pattern_syntax",  # repeat 0
        "invalid_pattern_syntax",  # brace not after a token
        "invalid_pattern_syntax",  # 'x' is not a token
        "invalid_pattern_syntax",  # unclosed repeat
    ]
    positions = [e.position for e in p.errors]
    assert positions == [5, 9, 13, 17]
    assert not p.valid and p.values == b""


def test_compile_rejects_lone_question_mark_and_single_hex_digit():
    p = compile_pattern("p", "? AB")
    assert [e.kind for e in p.errors] == ["invalid_pattern_syntax"]
    assert p.errors[0].position == 0
    p = compile_pattern("p", "A")
    assert not p.valid and p.errors[0].position == 0
    p = compile_pattern("p", "AB A ??")
    assert [e.position for e in p.errors] == [3]


def test_compile_rejects_bad_repeat_specifications():
    for syntax in ("AB{0}", "AB{}", "AB{-1}", "AB{ 2}", "AB{2 }", "AB{1x}"):
        p = compile_pattern("p", syntax)
        assert not p.valid, syntax
        assert p.errors[0].kind == "invalid_pattern_syntax", syntax


def test_compile_unclosed_repeat_consumes_rest_of_pattern():
    p = compile_pattern("p", "AB{3")
    assert [e.kind for e in p.errors] == ["invalid_pattern_syntax"]
    assert p.errors[0].position == 2
    assert p.values == b""


def test_compile_enforces_a_compiled_length_cap():
    p = compile_pattern("p", f"FF{{{MAX_PATTERN_BYTES + 1}}}")
    assert [e.kind for e in p.errors] == ["pattern_too_long"]
    p = compile_pattern("p", f"FF{{{MAX_PATTERN_BYTES}}}")
    assert p.valid and p.length == MAX_PATTERN_BYTES
    # cumulative expansion can exceed the cap even when no single repeat does
    half = MAX_PATTERN_BYTES // 2
    p = compile_pattern("p", f"FF{{{half + 1}}} AB{{{half + 1}}}")
    assert [e.kind for e in p.errors] == ["pattern_too_long"]


# ---------------------------------------------------------------------------
# scope validation
# ---------------------------------------------------------------------------

def test_scope_default_full_window_is_valid():
    assert validate_scope("p", 0, ADDR_SPAN, 1, 1) == []
    assert validate_scope("p", 16, 32, 4, 4) == []


def test_scope_rejects_empty_reversed_or_oversized_ranges():
    errors = validate_scope("p", 32, 32, 1, 1)
    assert [e.kind for e in errors] == ["invalid_pattern_range"]
    errors = validate_scope("p", 40, 32, 1, 1)
    assert len(errors) == 1 and errors[0].kind == "invalid_pattern_range"
    assert errors[0].start == 40 and errors[0].end == 32
    errors = validate_scope("p", 0, ADDR_SPAN + 1, 1, 1)
    assert [e.kind for e in errors] == ["invalid_pattern_range"]
    errors = validate_scope("p", -1, 16, 1, 1)
    assert [e.kind for e in errors] == ["invalid_pattern_range"]


def test_scope_range_must_hold_the_compiled_pattern():
    assert validate_scope("p", 0, 4, 1, 4) == []
    errors = validate_scope("p", 0, 4, 1, 5)
    assert [e.kind for e in errors] == ["invalid_pattern_range"]
    assert errors[0].start == 0 and errors[0].end == 4


def test_scope_rejects_non_positive_alignment():
    errors = validate_scope("p", 0, 16, 0, 1)
    assert [e.kind for e in errors] == ["invalid_pattern_alignment"]
    errors = validate_scope("p", 0, 16, -4, 1)
    assert [e.kind for e in errors] == ["invalid_pattern_alignment"]


def test_scope_can_report_range_and_alignment_errors_together():
    errors = validate_scope("p", 32, 12, 0, 1)
    assert {e.kind for e in errors} == {
        "invalid_pattern_range",
        "invalid_pattern_alignment",
    }


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def test_find_hits_basic_match_and_miss():
    segments = [seg(0, bytes.fromhex("AABBCC"))]
    hit = compile_pattern("p", "AA BB")
    miss = compile_pattern("p", "AA CC")
    assert find_hits(segments, hit, 0, ADDR_SPAN, 1) == [0]
    assert find_hits(segments, miss, 0, ADDR_SPAN, 1) == []


def test_find_hits_wildcards_match_any_byte():
    segments = [seg(0x100, bytes.fromhex("DEADBEEF"))]
    p = compile_pattern("p", "?? AD ?? EF")
    assert find_hits(segments, p, 0, ADDR_SPAN, 1) == [0x100]


def test_find_hits_does_not_cross_segment_holes():
    segments = [
        seg(0x100, bytes.fromhex("AABBCC")),
        seg(0x200, bytes.fromhex("AABBCC")),
    ]
    # "BB CC AA" would only match across the hole; it must be absent.
    spanning = compile_pattern("p", "BB CC AA")
    assert find_hits(segments, spanning, 0, ADDR_SPAN, 1) == []
    within = compile_pattern("p", "AA BB")
    assert find_hits(segments, within, 0, ADDR_SPAN, 1) == [0x100, 0x200]


def test_find_hits_reports_overlapping_matches():
    segments = [seg(0, bytes.fromhex("111111"))]
    p = compile_pattern("p", "11 11")
    assert find_hits(segments, p, 0, ADDR_SPAN, 1) == [0, 1]
    # fully-wild single byte matches every offset, overlapping by design
    p = compile_pattern("p", "??")
    assert find_hits(segments, p, 0, ADDR_SPAN, 1) == [0, 1, 2]


def test_find_hits_stepping_by_alignment():
    segments = [seg(0x1000, bytes.fromhex("AB00AB00AB00"))]
    p = compile_pattern("p", "AB")
    # AB at offsets 0, 2, 4; aligned to 2 -> all, aligned to 4 -> 0 and 4
    assert find_hits(segments, p, 0, ADDR_SPAN, 2) == [0x1000, 0x1002, 0x1004]
    assert find_hits(segments, p, 0, ADDR_SPAN, 4) == [0x1000, 0x1004]


def test_find_hits_is_clipped_to_the_half_open_window():
    segments = [seg(0x10, bytes.fromhex("AABB"))]
    p = compile_pattern("p", "AA BB")
    assert find_hits(segments, p, 0x10, 0x12, 1) == [0x10]
    # start and end are both excluded as match starts
    assert find_hits(segments, p, 0x11, ADDR_SPAN, 1) == []
    assert find_hits(segments, p, 0, 0x11, 1) == []
    # a match extending past window end is not reported
    assert find_hits(segments, p, 0x10, 0x11, 1) == []


def test_find_hits_with_alignment_relative_to_absolute_address():
    segments = [seg(0x1003, b"\xAB")]
    p = compile_pattern("p", "AB")
    assert find_hits(segments, p, 0, ADDR_SPAN, 1) == [0x1003]
    assert find_hits(segments, p, 0, ADDR_SPAN, 4) == []


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

def scan(payload: dict):
    return client.post("/api/v1/scan-byte-patterns", json=payload)


def test_endpoint_returns_unique_absent_and_ambiguous_statuses():
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "unique", "pattern": "21 46"},  # offset 0
                {"name": "absent", "pattern": "DE AD BE EF"},
                {"name": "many", "pattern": "??", "alignment": 1},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["hex_ok"] is True and body["patterns_valid"] is True
    assert body["total_bytes"] == 16
    assert body["entry_address"] is None
    by_name = {r["name"]: r for r in body["results"]}
    assert by_name["unique"]["status"] == "unique"
    assert by_name["unique"]["count"] == 1
    assert by_name["unique"]["hits"] == [
        {"pattern": "unique", "address": SEGMENT_START,
         "end": SEGMENT_START + 2, "length": 2}
    ]
    assert by_name["absent"]["status"] == "absent"
    assert by_name["absent"]["hits"] == []
    assert by_name["many"]["status"] == "ambiguous"
    assert by_name["many"]["count"] == 16
    assert body["total_hits"] == 17
    # first result order matches request order; default window echoed
    assert body["results"][0]["start"] == 0
    assert body["results"][0]["end"] == 0x100000000
    assert body["results"][0]["alignment"] == 1


def test_endpoint_flat_hits_are_sorted_by_address_then_pattern_order():
    # firmware: 21 46 01 36 01 21 47 01 36 00 7E FE 09 D2 19 01
    # "?? 36" matches offsets 2 (01 36) and 7 (01 36)
    # "01 ??" matches offsets 2 (01 36), 4 (01 21), 7 (01 36)
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "p_first", "pattern": "?? 36", "alignment": 1},
                {"name": "p_second", "pattern": "01 ??", "alignment": 1},
            ],
        }
    )
    body = resp.json()
    assert body["ok"] is True
    addresses = [(h["address"], h["pattern"]) for h in body["hits"]]
    assert addresses == sorted(addresses)
    first_hits = [h["address"] for h in body["hits"] if h["pattern"] == "p_first"]
    second_hits = [h["address"] for h in body["hits"] if h["pattern"] == "p_second"]
    assert first_hits == sorted(first_hits) and second_hits == sorted(second_hits)
    shared = sorted(set(first_hits) & set(second_hits))
    assert shared == [SEGMENT_START + 2, SEGMENT_START + 7]
    for address in shared:
        indices = [
            i for i, h in enumerate(body["hits"]) if h["address"] == address
        ]
        assert body["hits"][indices[0]]["pattern"] == "p_first"
        assert body["hits"][indices[1]]["pattern"] == "p_second"
    assert [h["address"] for h in body["hits"]] == [
        SEGMENT_START + 2,
        SEGMENT_START + 2,
        SEGMENT_START + 4,
        SEGMENT_START + 7,
        SEGMENT_START + 7,
    ]


def test_endpoint_expected_count_drives_count_mismatch():
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "ok", "pattern": "21 46", "expected_count": 1},
                {"name": "mismatch", "pattern": "21 46", "expected_count": 3},
                {"name": "absent_ok", "pattern": "99", "expected_count": 0},
            ],
        }
    )
    body = resp.json()
    by_name = {r["name"]: r for r in body["results"]}
    assert by_name["ok"]["count_mismatch"] is False
    assert by_name["mismatch"]["count_mismatch"] is True
    assert by_name["absent_ok"]["count_mismatch"] is False
    # a mismatch is a successful scan, not a request error
    assert body["ok"] is True and body["pattern_errors"] == []


def test_endpoint_repeats_and_wildcards_scan_over_real_firmware():
    parsed = parse_intel_hex(SINGLE_SEGMENT_HEX)
    # firmware starts: 21 46 01 36 01 21 47 01 36 ... ; 01 ?? fits at 2, 4, 7
    p = compile_pattern("p", "01 ??")
    hits = find_hits(parsed.segments, p, 0, ADDR_SPAN, 1)
    assert hits == [SEGMENT_START + 2, SEGMENT_START + 4, SEGMENT_START + 7]
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [{"name": "p", "pattern": "01 ??", "alignment": 1}],
        }
    )
    body = resp.json()
    assert body["results"][0]["status"] == "ambiguous"
    assert [h["address"] for h in body["hits"]] == hits


def test_endpoint_does_not_match_across_hex_gap():
    hex_text = "\n".join(
        [
            linear_base(0x0800),
            data_rec(0x0000, bytes.fromhex("AABBCC")),
            data_rec(0x1000, bytes.fromhex("AABBCC")),
            EOF,
        ]
    )
    resp = scan(
        {
            "hex_text": hex_text,
            "patterns": [
                {"name": "span", "pattern": "BB CC AA"},
                {"name": "within", "pattern": "AA BB"},
            ],
        }
    )
    body = resp.json()
    assert body["ok"] is True
    by_name = {r["name"]: r for r in body["results"]}
    assert by_name["span"]["status"] == "absent"
    assert [h["address"] for h in by_name["within"]["hits"]] == [
        0x08000000,
        0x08001000,
    ]


def test_endpoint_window_and_alignment_filter_matches():
    # byte 0x36 sits at offsets 3 and 8 (both odd/even absolute addresses)
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "in_window", "pattern": "36",
                 "start": SEGMENT_START + 3, "end": SEGMENT_START + 4},
                {"name": "before_window", "pattern": "36",
                 "start": SEGMENT_START + 4, "end": SEGMENT_START + 8},
                {"name": "aligned4", "pattern": "36",
                 "start": None, "end": None, "alignment": 4},
            ],
        }
    )
    body = resp.json()
    assert body["ok"] is True
    by_name = {r["name"]: r for r in body["results"]}
    assert by_name["in_window"]["status"] == "unique"
    assert by_name["before_window"]["status"] == "absent"
    # offset 8 address is divisible by 4, offset 3 address is not
    assert [h["address"] for h in by_name["aligned4"]["hits"]] == [
        SEGMENT_START + 8
    ]


def test_endpoint_aggregates_hex_and_pattern_diagnostics_without_scanning():
    resp = scan(
        {
            "hex_text": ":10000000214601360121470136007EFE09D2190100\n",  # bad cksum, no EOF
            "patterns": [
                {"name": "bad1", "pattern": "AB{0}"},
                {"name": "bad2", "pattern": "? "},
                {"name": "fine", "pattern": "AB"},
            ],
        }
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["hex_ok"] is False
    assert body["patterns_valid"] is False
    assert body["results"] == [] and body["hits"] == [] and body["total_hits"] == 0
    assert body["error_count"] >= 2  # checksum error + missing EOF
    kinds = [(e["pattern"], e["kind"]) for e in body["pattern_errors"]]
    assert ("bad1", "invalid_pattern_syntax") in kinds
    assert ("bad2", "invalid_pattern_syntax") in kinds
    assert all(e["pattern"] != "fine" for e in body["pattern_errors"])
    assert len(body["diagnostics"]) >= 2


def test_endpoint_reports_invalid_ranges_and_alignments_together():
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "r", "pattern": "AB CD", "start": 100, "end": 101},
                {"name": "a", "pattern": "AB CD FF FF",
                 "start": SEGMENT_START, "end": SEGMENT_START + 2},
            ],
        }
    )
    body = resp.json()
    assert body["ok"] is False and body["hex_ok"] is True
    assert body["patterns_valid"] is False
    assert body["results"] == [] and body["hits"] == []
    by_pattern = {}
    for e in body["pattern_errors"]:
        by_pattern.setdefault(e["pattern"], []).append(e["kind"])
    assert by_pattern["r"] == ["invalid_pattern_range"]
    assert by_pattern["a"] == ["invalid_pattern_range"]


def test_endpoint_position_points_into_pattern_syntax():
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [{"name": "p", "pattern": "AB {3}"}],
        }
    )
    errors = resp.json()["pattern_errors"]
    assert len(errors) == 1
    assert errors[0]["kind"] == "invalid_pattern_syntax"
    assert errors[0]["position"] == 3


def test_endpoint_requires_at_least_one_pattern_and_unique_names():
    resp = scan({"hex_text": SINGLE_SEGMENT_HEX, "patterns": []})
    assert resp.status_code == 422
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [
                {"name": "dup", "pattern": "AB"},
                {"name": "dup", "pattern": "CD"},
            ],
        }
    )
    assert resp.status_code == 422
    assert "dup" in resp.text


def test_endpoint_schema_rejects_bad_numeric_parameters():
    base = {"hex_text": SINGLE_SEGMENT_HEX}
    for payload in [
        {**base, "patterns": [{"name": "p", "pattern": "AB", "start": -1}]},
        {**base, "patterns": [{"name": "p", "pattern": "AB", "end": -1}]},
        {**base, "patterns": [{"name": "p", "pattern": "AB", "alignment": 0}]},
        {**base, "patterns": [{"name": "p", "pattern": "AB", "expected_count": -1}]},
        {**base, "patterns": [{"name": "", "pattern": "AB"}]},
    ]:
        assert scan(payload).status_code == 422, payload


def test_endpoint_invalid_hex_alone_still_compiles_patterns_but_does_not_scan():
    resp = scan(
        {
            "hex_text": "not a hex file",
            "patterns": [{"name": "p", "pattern": "AB"}],
        }
    )
    body = resp.json()
    assert body["ok"] is False and body["hex_ok"] is False
    assert body["patterns_valid"] is True and body["pattern_errors"] == []
    assert body["results"] == [] and body["hits"] == []


def test_endpoint_full_success_response_shape():
    resp = scan(
        {
            "hex_text": SINGLE_SEGMENT_HEX,
            "patterns": [{"name": "p", "pattern": "2146{1}", "alignment": 2,
                          "expected_count": 1}],
        }
    )
    body = resp.json()
    result = body["results"][0]
    assert result == {
        "name": "p",
        "pattern": "2146{1}",
        "status": "unique",
        "count": 1,
        "expected_count": 1,
        "count_mismatch": False,
        "start": 0,
        "end": 0x100000000,
        "alignment": 2,
        "hits": [
            {"pattern": "p", "address": SEGMENT_START,
             "end": SEGMENT_START + 2, "length": 2}
        ],
    }
    assert body["hits"][0]["address"] == SEGMENT_START
    assert body["pattern_count"] == 1 and body["total_hits"] == 1
