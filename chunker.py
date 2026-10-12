import re, json

MAX_CHARS = 1800   # about 450 tokens
MIN_CHARS = 300

def split_sections(md):
    """Yield (heading_path, text) per heading section, ignoring '#' inside code fences."""
    path, buf, in_code = [], [], False
    def flush():
        text = "\n".join(buf).strip()
        if text:
            yield (" > ".join(h for _, h in path), text)
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            yield from flush(); buf.clear()
            level = len(m.group(1))
            path = [(l, h) for l, h in path if l < level] + [(level, m.group(2).strip())]
        else:
            buf.append(line)
    yield from flush()

def split_long(text):
    """Split at blank lines, keeping code blocks intact."""
    parts, cur = [], ""
    for block in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(block) > MAX_CHARS:
            parts.append(cur.strip()); cur = ""
        cur += block + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts

chunks = []
for path, text in split_sections(open("netze.clean.md", encoding="utf-8").read()):
    for part in split_long(text) if len(text) > MAX_CHARS else [text]:
        if chunks and len(part) < MIN_CHARS and chunks[-1]["path"] == path:
            chunks[-1]["text"] += "\n\n" + part          # merge tiny leftovers
        else:
            chunks.append({"id": len(chunks), "path": path, "text": part})

for c in chunks:
    c["embed_text"] = f"passage: {c['path']}\n{c['text']}"   # e5 prefix + heading context

json.dump(chunks, open("chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(chunks), "chunks")