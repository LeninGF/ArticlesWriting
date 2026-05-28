#!/usr/bin/env python3
"""
make-final.py — Strip \\added, \\deleted, \\replaced from a LaTeX file.

Two modes
---------
Default (no --inplace flag):
    Produces preeclampsia-ml-hia-FINAL.tex — completely clean, no traces.

--inplace  (or pass an explicit output path as 2nd argument):
    Produces a self-contained .tex where:
      \\added[id]{TEXT}          →  TEXT
      \\deleted[id]{TEXT}        →  %% [DELETED id]: TEXT   (commented out)
      \\replaced[id]{NEW}{OLD}   →  NEW
                                    %% [OLD id]: OLD        (old text commented)
    The \\usepackage{changes} and \\definechangesauthor lines are commented out.
    Result compiles without the changes package; old text is preserved in comments.

Usage:
    python3 make-final.py preeclampsia-ml-hia.tex
        → preeclampsia-ml-hia-FINAL.tex  (fully stripped)

    python3 make-final.py preeclampsia-ml-hia.tex preeclampsia-ml-hia.tex
        → overwrites the source with the commented-old-text version
"""

import re
import sys
from pathlib import Path


def extract_brace_argument(text, start):
    """
    Given `text` and the index of an opening '{' at `start`,
    return (content, end_index) where end_index is the index
    AFTER the closing '}'.  Handles nested braces.
    """
    assert text[start] == '{'
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == '\\':          # skip escaped character
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError(f"Unmatched brace starting at position {start}")


def skip_optional_arg(text, pos):
    """
    If text[pos] == '[', skip over the optional [...] argument
    (brace-aware) and return the position after it.
    Otherwise return pos unchanged.
    """
    if pos >= len(text) or text[pos] != '[':
        return pos
    depth = 0
    i = pos
    while i < len(text):
        c = text[i]
        if c == '\\':
            i += 2
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return pos  # malformed, give up


def comment_out(text, label):
    """
    Wrap `text` as a LaTeX comment block, prefixing each line with '%% '.
    `label` is prepended to the first line, e.g. '[DELETED R1]'.
    """
    lines = text.split('\n')
    # Remove leading/trailing blank lines inside the block
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ''
    out = [f'%% {label}: {lines[0]}']
    for ln in lines[1:]:
        out.append(f'%% {ln}')
    return '\n'.join(out)


def strip_changes(src: str, comment_deleted: bool = False) -> str:
    result = []
    i = 0
    n = len(src)

    # Regex to find any of the three change commands
    cmd_re = re.compile(
        r'\\(added|deleted|replaced)\b'
    )

    while i < n:
        m = cmd_re.search(src, i)
        if m is None:
            result.append(src[i:])
            break

        # Append everything up to the command
        result.append(src[i:m.start()])
        cmd = m.group(1)
        pos = m.end()

        # Capture optional [id=...] for use in comment labels
        opt_start = pos
        pos = skip_optional_arg(src, pos)
        opt_raw = src[opt_start:pos].strip('[] ').replace('id=', '').strip() if pos > opt_start else ''
        label_sfx = f' {opt_raw}' if opt_raw else ''

        if cmd == 'added':
            # \added[opt]{TEXT}  →  TEXT
            if pos < n and src[pos] == '{':
                content, pos = extract_brace_argument(src, pos)
                result.append(content)

        elif cmd == 'deleted':
            # \deleted[opt]{TEXT}  →  (nothing) or %% [DELETED id]: ...
            if pos < n and src[pos] == '{':
                content, pos = extract_brace_argument(src, pos)
                if comment_deleted and content.strip():
                    result.append('\n' + comment_out(content, f'[DELETED{label_sfx}]'))

        elif cmd == 'replaced':
            # \replaced[opt]{NEW}{OLD}  →  NEW  (OLD commented if requested)
            if pos < n and src[pos] == '{':
                new_text, pos = extract_brace_argument(src, pos)
                old_text = ''
                if pos < n and src[pos] == '{':
                    old_text, pos = extract_brace_argument(src, pos)
                result.append(new_text)
                if comment_deleted and old_text.strip():
                    result.append('\n' + comment_out(old_text, f'[OLD{label_sfx}]'))

        i = pos

    return ''.join(result)


def patch_preamble(src: str) -> str:
    """Replace the changes package line and comment out \definechangesauthor."""
    # Disable the markup version, remove [final] line, add a plain import
    src = re.sub(
        r'\\usepackage\[final\]\{changes\}',
        r'% \\usepackage[final]{changes}  % stripped by make-final.py',
        src
    )
    src = re.sub(
        r'\\usepackage\[markup=[^\]]+\]\{changes\}',
        r'% stripped by make-final.py',
        src
    )
    src = re.sub(
        r'\\definechangesauthor\[',
        r'% \\definechangesauthor[',
        src
    )
    return src


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 make-final.py <input.tex> [output.tex]")
        print("  1 arg  → produces <stem>-FINAL.tex  (fully clean, no comments)")
        print("  2 args → writes to output.tex with old/deleted text in %% comments")
        sys.exit(1)

    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"File not found: {inp}")
        sys.exit(1)

    # Two-arg form: output path given → comment-preserved mode
    comment_mode = len(sys.argv) >= 3
    out = Path(sys.argv[2]) if comment_mode else inp.with_stem(inp.stem + '-FINAL')

    src = inp.read_text(encoding='utf-8')
    src = patch_preamble(src)
    src = strip_changes(src, comment_deleted=comment_mode)

    out.write_text(src, encoding='utf-8')
    mode_label = 'comment-preserved' if comment_mode else 'fully stripped'
    print(f"Written ({mode_label}): {out}")
    print("Now run:  latexmk -pdf " + str(out))


if __name__ == '__main__':
    main()
