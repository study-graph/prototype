import re
from collections import Counter

SRC, DST = "netze.md", "netze.clean.md"
MIN_REPEAT = 8          # a line repeated this often across the deck is chrome, not content


def flatten_row(line):
    """A markdown table row -> its non-empty cells joined by spaces. '' if separator."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(re.fullmatch(r"-{2,}|:?-+:?|", c) for c in cells):
        return ""                       # separator row (| --- | --- |)
    return " ".join(c for c in cells if c)


def normalize(line):
    """Plain-text content of a line, flattening tables and ragged pdf whitespace."""
    s = line.strip()
    if s.startswith("|"):
        s = flatten_row(s)
    return re.sub(r"\s+", " ", s).strip()


def looks_like_nav(line):
    """Nav bars are long runs of capitalised section words with no sentence punctuation."""
    words = line.split()
    return len(words) >= 5 and not re.search(r"[.:!?]", line) \
        and sum(w[:1].isupper() for w in words) >= len(words) - 1


def main():
    lines = [normalize(l) for l in open(SRC, encoding="utf-8")]

    # Chrome = any non-empty line repeated across many slides (footers, nav bars, author).
    freq = Counter(l for l in lines if l)
    chrome = {l for l, n in freq.items() if n >= MIN_REPEAT}
    navs = {l for l in chrome if looks_like_nav(l)}

    out, slide_started = [], False
    for line in lines:
        if not line:
            continue
        if line in navs:
            slide_started = True        # next real content line starts a new slide
            if out and out[-1] != "":
                out.append("")
            continue
        if line in chrome:              # footer / author / repeated boilerplate
            continue
        if slide_started:
            out.append(f"## {line}")     # promote slide title to a heading
            slide_started = False
        else:
            out.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    open(DST, "w", encoding="utf-8").write(text)
    print(f"{DST}: {text.count(chr(10))} lines, {text.count('## ')} headings, "
          f"{len(navs)} nav bars, {len(chrome)} chrome lines dropped")


if __name__ == "__main__":
    main()
