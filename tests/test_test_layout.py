"""Policy test: every Python test labels its Arrange, Act, and Assert phases.

Each test function has exactly one ``# Act`` (or ``# Act & Assert`` when the check wraps the
action), at most one ``# Arrange`` before it, and an ``# Assert`` after a plain ``# Act``.
"""

import ast
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent
_ACT_LABELS = ("# Act", "# Act & Assert")
_ALL_LABELS = ("# Arrange", *_ACT_LABELS, "# Assert")


def _labels(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the phase labels inside one function, in order."""
    body = lines[node.lineno : node.end_lineno]
    return [line.strip() for line in body if line.strip() in _ALL_LABELS]


def _layout_problem(labels: list[str]) -> str | None:
    """Return why a label sequence breaks the layout, or ``None`` when it is valid."""
    acts = [label for label in labels if label in _ACT_LABELS]
    if len(acts) != 1:
        return f"expected one act label, found {len(acts)}"
    act_index = labels.index(acts[0])
    if labels.count("# Arrange") > 1 or labels.count("# Assert") > 1:
        return "repeated arrange or assert label"
    if "# Arrange" in labels and labels.index("# Arrange") > act_index:
        return "arrange after act"
    if "# Assert" in labels and labels.index("# Assert") < act_index:
        return "assert before act"
    if acts[0] == "# Act" and "# Assert" not in labels:
        return "act without assert"
    return None


def test_every_test_function_labels_its_phases() -> None:
    # Arrange
    paths = sorted(_TESTS_ROOT.rglob("test_*.py"))

    # Act
    problems = [
        f"{path.name}:{node.lineno} {node.name}: {problem}"
        for path in paths
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
        and (
            problem := _layout_problem(_labels(path.read_text(encoding="utf-8").splitlines(), node))
        )
    ]

    # Assert
    assert problems == []


def test_layout_problem_rejects_a_test_with_two_actions() -> None:
    # Act
    problem = _layout_problem(["# Arrange", "# Act", "# Assert", "# Act", "# Assert"])

    # Assert
    assert problem is not None


def test_layout_problem_accepts_act_and_assert_without_arrange() -> None:
    # Act
    problem = _layout_problem(["# Act & Assert"])

    # Assert
    assert problem is None
