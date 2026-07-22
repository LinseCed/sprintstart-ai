"""Classify a source file as test/fixture material by its path — language-agnostic.

The buddy grounds answers in retrieved files, and some of those files are tests,
fixtures, mocks or sample data rather than the team's real documentation or
process. The buddy should say so instead of quoting them as authoritative — e.g.
a "Code Review SLA" that actually lives in a `tests/.../demo-corpus/` fixture is
example data, not the team's policy.

This recognises such files the way every ecosystem marks them: a test directory
somewhere in the path, or a conventional test/spec suffix on the file name. It
keys only on naming conventions, so it works for any repo, language or project
with no configuration. It is deliberately conservative — filename patterns are
tied to known code extensions so data and config files (``openapi_spec.yaml``,
``manifest.json``) are never misread as tests.
"""

import re

# Path *directory* components that mark everything beneath them as test/fixture
# material, across ecosystems (pytest, jest, go, maven/gradle, rspec, ...).
_TEST_DIR_SEGMENTS = frozenset(
    {
        "test",
        "tests",
        "__tests__",
        "__mocks__",
        "__snapshots__",
        "snapshots",
        "testdata",
        "test-data",
        "test_data",
        "fixture",
        "fixtures",
        "mocks",
        "e2e",
        "testing",
    }
)

# Extensions that carry executable test code, so a test/spec suffix on one is a
# real test rather than a coincidentally-named document or data file.
_CODE_EXT = (
    r"(?:py|rb|go|js|jsx|ts|tsx|mjs|cjs|java|kt|kts|scala|groovy|cs|swift|php|"
    r"rs|dart|ex|exs|clj|c|cc|cpp|h|hpp|m|mm)"
)

# File *names* that are tests by convention even when they sit next to the code
# they exercise (Go's ``_test.go``, co-located ``foo.test.ts``, xUnit classes).
_TEST_FILENAME_RE = re.compile(
    r"^conftest\.py$"
    r"|^test_.+\.(?:py|rb)$"
    rf"|.+_test\.{_CODE_EXT}$"
    r"|.+_spec\.rb$"
    r"|.+\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$"
    r"|.+(?:Test|Tests|Spec|IT)\.(?:java|kt|kts|scala|groovy|cs|swift|php)$"
)


def is_test_source(path: str | None) -> bool:
    """True when ``path`` names a test, fixture, mock or sample-data file.

    Recognises a test directory anywhere in the path, or a conventional test/spec
    filename suffix. Case-insensitive on directory names; filename suffixes keep
    their conventional casing (``FooTest.java`` is a test, ``footest.md`` is not).
    """
    if not path:
        return False
    parts = [segment for segment in path.replace("\\", "/").split("/") if segment]
    if not parts:
        return False
    if any(segment.lower() in _TEST_DIR_SEGMENTS for segment in parts[:-1]):
        return True
    return _TEST_FILENAME_RE.match(parts[-1]) is not None


__all__ = ["is_test_source"]
