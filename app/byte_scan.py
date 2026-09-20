"""Byte-pattern scanning over assembled firmware segments.

A pattern is a whitespace-separated sequence of one-byte tokens:

* two uppercase/lowercase hexadecimal digits (``"AB"``) — a fixed byte;
* ``"??"`` — a single-byte wildcard matching any value;
* either token may be immediately followed by ``"{n}"`` repeating it
  ``n`` times (``n`` a positive decimal integer), e.g. ``"FF{16}"`` or
  ``"??{4} AB CD"``.

Each pattern is compiled into parallel ``values``/``masks`` byte
sequences: a fixed byte compiles to ``(value, 0xFF)`` and a wildcard to
``(0, 0x00)``. Matching then tests
``(firmware_byte & mask) == value`` for every position.

Matching runs only inside the parser's maximal contiguous firmware
segments, so a match can never span an inter-segment address hole.
Candidates may be restricted to a half-open address window and to
start addresses divisible by an alignment value; overlapping matches of
the same pattern are all reported (the candidate step equals the
alignment, so unaligned starts are never considered).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hex_parser import Segment

ADDR_SPAN = 1 << 32  # exclusive upper bound of the 32-bit address space

# Hard cap on the compiled (post-expansion) pattern length so a repeat
# count cannot blow up memory or the scan.
MAX_PATTERN_BYTES = 65536

_HEX_DIGITS = set("0123456789abcdefABCDEF")
_WHITESPACE = " \t\r\n\v\f"


@dataclass
class PatternError:
    """A pattern-level problem that prevents scanning.

    ``kind`` is one of ``invalid_pattern_syntax``, ``pattern_too_long``,
    ``invalid_pattern_range`` or ``invalid_pattern_alignment``.
    ``position`` is the 0-based character offset inside the pattern
    syntax for syntax problems.
    """

    kind: str
    pattern: str | None
    message: str
    position: int | None = None
    start: int | None = None
    end: int | None = None
    alignment: int | None = None


@dataclass
class CompiledPattern:
    """The value/mask representation of one named pattern."""

    name: str
    syntax: str
    values: bytes = b""
    masks: bytes = b""
    errors: list[PatternError] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Compiled byte count (0 when compilation failed)."""
        return len(self.values)

    @property
    def valid(self) -> bool:
        """True when no syntax/expansion error was reported."""
        return not self.errors


def _fmt(addr: int) -> str:
    return f"0x{addr:08X}" if isinstance(addr, int) and addr >= 0 else repr(addr)


def _syntax_error(
    name: str | None, message: str, position: int | None
) -> PatternError:
    return PatternError(
        kind="invalid_pattern_syntax",
        pattern=name,
        message=message,
        position=position,
    )


def compile_pattern(name: str, syntax: str) -> CompiledPattern:
    """Compile pattern ``syntax`` into value/mask byte sequences.

    Every syntax problem is collected (a malformed token does not stop
    the parser from reporting later problems such as an unclosed
    repeat); the returned pattern is valid only when the error list is
    empty.
    """
    errors: list[PatternError] = []
    values: list[int] = []
    masks: list[int] = []

    def add(message: str, position: int | None) -> None:
        errors.append(_syntax_error(name, message, position))

    n = len(syntax)
    i = 0

    def skip_ws() -> None:
        nonlocal i
        while i < n and syntax[i] in _WHITESPACE:
            i += 1

    while True:
        skip_ws()
        if i >= n:
            break

        if syntax[i] == "{":
            add(
                "'{' must immediately follow a byte token (two hex digits or '??')",
                i,
            )
            close = syntax.find("}", i + 1)
            i = n if close == -1 else close + 1
            continue

        token_start = i
        if syntax[i] == "?":
            if i + 1 >= n or syntax[i + 1] != "?":
                add(
                    "a wildcard must be written '??' (a lone '?' is not a token)",
                    i,
                )
                i += 1
                continue
            value, mask = 0, 0x00
            i += 2
        else:
            first = syntax[i]
            second = syntax[i + 1] if i + 1 < n else ""
            if first not in _HEX_DIGITS or second not in _HEX_DIGITS:
                add(
                    "expected a byte token: two hexadecimal digits or '??'",
                    i,
                )
                i += 1
                continue
            value, mask = int(first + second, 16), 0xFF
            i += 2

        repeat = 1
        repeat_error: PatternError | None = None
        if i < n and syntax[i] == "{":
            brace_pos = i
            close = syntax.find("}", i + 1)
            if close == -1:
                add(
                    "unclosed repeat: expected '{n}' with a positive decimal "
                    "number immediately after the token",
                    brace_pos,
                )
                i = n
                continue
            inner = syntax[i + 1 : close]
            if not re.fullmatch(r"[0-9]+", inner):
                repeat_error = _syntax_error(
                    name,
                    "repeat '{n}' must contain a positive decimal number of "
                    f"repetitions, with no spaces (got '{inner}')",
                    brace_pos,
                )
            else:
                repeat = int(inner)
                if repeat == 0:
                    repeat_error = _syntax_error(
                        name,
                        "repeat count must be at least 1 (got '{0}')",
                        brace_pos,
                    )
            i = close + 1

        if repeat_error is not None:
            errors.append(repeat_error)
            continue

        if repeat > MAX_PATTERN_BYTES:
            errors.append(
                PatternError(
                    kind="pattern_too_long",
                    pattern=name,
                    message=(
                        f"repeat count {repeat} alone exceeds the {MAX_PATTERN_BYTES} "
                        "byte limit for a compiled pattern"
                    ),
                    position=token_start,
                )
            )
            break

        values.extend([value] * repeat)
        masks.extend([mask] * repeat)
        if len(values) > MAX_PATTERN_BYTES:
            errors.append(
                PatternError(
                    kind="pattern_too_long",
                    pattern=name,
                    message=(
                        f"compiled pattern would be at least {len(values)} "
                        f"byte(s); at most {MAX_PATTERN_BYTES} bytes are allowed"
                    ),
                    position=token_start,
                )
            )
            break

    if not values and not errors:
        add(
            "pattern syntax is empty; expected one or more byte tokens "
            "(two hex digits or '??', each optionally followed by '{n}')",
            0 if n else None,
        )

    return CompiledPattern(
        name=name,
        syntax=syntax,
        values=bytes(values) if not errors else b"",
        masks=bytes(masks) if not errors else b"",
        errors=errors,
    )


def validate_scope(
    name: str,
    start: int,
    end: int,
    alignment: int,
    pattern_length: int,
) -> list[PatternError]:
    """Validate a per-pattern half-open window and alignment.

    The window must be a non-empty half-open interval inside the 32-bit
    address space large enough to contain the (compiled) pattern, and
    the alignment must be a positive integer.
    """
    errors: list[PatternError] = []

    if not isinstance(alignment, int) or alignment < 1:
        errors.append(
            PatternError(
                kind="invalid_pattern_alignment",
                pattern=name,
                message=(
                    f"pattern '{name}': alignment must be a positive integer "
                    f"(got {alignment!r})"
                ),
                alignment=alignment if isinstance(alignment, int) else None,
            )
        )

    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < 0
        or end > ADDR_SPAN
        or start >= end
    ):
        errors.append(
            PatternError(
                kind="invalid_pattern_range",
                pattern=name,
                message=(
                    f"pattern '{name}': range [{_fmt(start)}, {_fmt(end)}) must "
                    "be a non-empty half-open interval within the 32-bit "
                    "address space (0 <= start < end <= 0x100000000)"
                ),
                start=start if isinstance(start, int) else None,
                end=end if isinstance(end, int) else None,
            )
        )
    elif end - start < pattern_length:
        errors.append(
            PatternError(
                kind="invalid_pattern_range",
                pattern=name,
                message=(
                    f"pattern '{name}': compiled pattern is {pattern_length} "
                    f"byte(s) long but the range [{_fmt(start)}, {_fmt(end)}) "
                    f"spans only {end - start} byte(s), so it can never match"
                ),
                start=start,
                end=end,
            )
        )

    return errors


def find_hits(
    segments: list[Segment],
    pattern: CompiledPattern,
    start: int,
    end: int,
    alignment: int,
) -> list[int]:
    """Return every matching start address for one compiled pattern.

    Only windows fully contained in a single contiguous segment are
    considered, so matches cannot cross address holes. Candidates step
    by ``alignment`` (unaligned starts never match) and step one
    alignment unit at a time, so aligned, overlapping matches are all
    reported. Segments are visited in address order; returned addresses
    are therefore ascending.
    """
    length = pattern.length
    if length == 0:
        return []

    values = pattern.values
    masks = pattern.masks
    addresses: list[int] = []

    for segment in sorted(segments, key=lambda s: s.start):
        window_low = max(segment.start, start)
        window_high = min(segment.end, end) - length
        if window_low > window_high:
            continue

        remainder = window_low % alignment
        first = window_low if remainder == 0 else window_low + (alignment - remainder)
        for address in range(first, window_high + 1, alignment):
            offset = address - segment.start
            for pos in range(length):
                if segment.data[offset + pos] & masks[pos] != values[pos]:
                    break
            else:
                addresses.append(address)

    return addresses
