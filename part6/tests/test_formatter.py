from formatters import Example, format_example, format_prompt

def test_template_contains_markers():
    ex = Example("Say hi","Hello!")
    s = format_example(ex)
    assert "### Instruction:" in s and "### Response:" in s
    p = format_prompt("Explain transformers.")
    assert p.endswith("### Response:\n") or p.endswith("### Response:\n</s>")