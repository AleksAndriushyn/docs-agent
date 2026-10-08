from agents.citations import Citations


def test_same_ref_keeps_its_number():
    c = Citations()
    assert c.add("a.rst:1-10") == 1
    assert c.add("b.rst:5-9") == 2
    assert c.add("a.rst:1-10") == 1


def test_footer_lists_cited_in_order_and_ignores_unknown():
    c = Citations()
    c.add("a.rst:1-10"), c.add("b.rst:5-9"), c.add("c.rst:1-2")
    footer = c.footer("X [3]. Y [1][3]. Z [7].")
    assert footer == "\n\nSources:\n[3] c.rst:1-2\n[1] a.rst:1-10"


def test_footer_without_citations_lists_retrieved():
    c = Citations()
    c.add("a.rst:1-10")
    assert "Retrieved (not cited)" in c.footer("no refs here")
    c.reset()
    assert c.footer("anything") == ""


def test_remap_into_caller_numbering():
    sub, sup = Citations(), Citations()
    sup.add("x.rst:1-2")
    sub.add("a.rst:1-10"), sub.add("b.rst:5-9")
    assert sub.remap("A [2]. B [1]. C [9].", sup) == "A [2]. B [3]. C ."
    assert sup.refs == ["x.rst:1-2", "b.rst:5-9", "a.rst:1-10"]
