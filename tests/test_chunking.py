from rag.chunking import TARGET_CHARS, chunk
from rag.loaders import Document


def covers_all_lines(chunks, n_lines):
    covered = set()
    for c in chunks:
        covered.update(range(c.line_start, c.line_end + 1))
    return covered == set(range(1, n_lines + 1))


def test_rst_splits_by_titles():
    body = "text line\n" * 80
    text = f"Title\n=====\n\n{body}\nSection A\n---------\n{body}\nSection B\n---------\n{body}"
    chunks = chunk(Document("a.rst", "rst", text))
    assert any("Section A" in c.symbol for c in chunks)
    assert any(c.text.startswith("Section B") for c in chunks)
    assert covers_all_lines(chunks, len(text.splitlines()))


def test_markdown_headings_and_merge_small():
    text = "# A\nshort\n## B\nshort\n## C\nshort"
    chunks = chunk(Document("a.md", "markdown", text))
    assert len(chunks) == 1  # small sections merge
    assert chunks[0].symbol == "A, B, C"


def test_python_big_function_kept_whole_with_name():
    filler = "\n".join(f"    value_{i} = compute_something({i})" for i in range(30))
    text = f"import os\n\n\ndef first():\n{filler}\n\n\ndef second():\n{filler}\n"
    chunks = chunk(Document("m.py", "python", text))
    assert [c.symbol for c in chunks] == ["first", "second"]
    second = chunks[1]
    assert text.splitlines()[second.line_start - 1].startswith("def second")
    assert covers_all_lines(chunks, len(text.splitlines()))


def test_python_big_class_split_with_context():
    method = "\n".join(f"        value_{i} = compute_something({i})" for i in range(30))
    text = "class Big:\n" + "".join(f"    def m{j}(self):\n{method}\n\n" for j in range(3))
    chunks = chunk(Document("c.py", "python", text))
    assert len(chunks) >= 3
    assert all(c.context == "class Big:" for c in chunks[1:])
    assert {"m1", "m2"} <= {s for c in chunks for s in c.symbol.split(", ")}


def test_csharp_methods():
    method = "\n".join(f"            var v{i} = {i};" for i in range(40))
    text = ("namespace MP.Plugins {\n  public class Foo {\n"
            + "".join(f"    public void Run{j}() {{\n{method}\n    }}\n" for j in range(3))
            + "  }\n}\n")
    chunks = chunk(Document("f.cs", "csharp", text))
    names = {s for c in chunks for s in c.symbol.split(", ")}
    assert {"Run0", "Run1", "Run2"} <= names
    assert all(len(c.text) <= TARGET_CHARS * 2 for c in chunks)


def test_long_text_uses_windows():
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(300))
    chunks = chunk(Document("a.txt", "text", text))
    assert len(chunks) > 3
    assert covers_all_lines(chunks, 300)


def test_long_lines_are_capped_by_chars():
    text = "\n".join("  #1, `LINK <https://example.com/" + "x" * 150 + ">`_" for _ in range(80))
    chunks = chunk(Document("t.rst", "rst", text))
    assert all(len(c.text) <= TARGET_CHARS for c in chunks)
    assert covers_all_lines(chunks, 80)


def test_glob_regex():
    from rag.loaders import glob_regex
    root_cs, tree, deep = glob_regex("*.cs"), glob_regex("GCSViews/**"), glob_regex("**/*.Designer.cs")
    assert root_cs.match("Program.cs") and not root_cs.match("GCSViews/FlightData.cs")
    assert tree.match("GCSViews/a/b.cs") and not tree.match("Controls/x.cs")
    assert deep.match("Form.Designer.cs") and deep.match("GCSViews/FlightData.Designer.cs")
    assert not deep.match("GCSViews/FlightData.cs")
