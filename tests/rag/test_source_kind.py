"""Tests for the language-agnostic test/fixture source classifier."""

import pytest

from rag.source_kind import is_test_source


@pytest.mark.parametrize(
    "path",
    [
        # The case that started this: a fixture under a tests/ tree.
        "tests/rag/demo-corpus/process.md",
        "tests/insights/test_faq.py",
        # Test directories, various ecosystems and depths.
        "test/models/user_model.rb",
        "src/__tests__/app.js",
        "app/__mocks__/api.ts",
        "backend/src/testData/sample.json",
        "e2e/checkout.spec.ts",
        "src/components/__snapshots__/Button.snap",
        "web/fixtures/users.yaml",
        # Conventional filenames sitting next to the code they exercise.
        "internal/server/server_test.go",
        "pkg/util/util_test.py",
        "src/components/Button.test.tsx",
        "src/components/Button.spec.js",
        "src/main/java/com/acme/UserServiceTest.java",
        "src/test/kotlin/com/acme/UserServiceSpec.kt",
        "it/com/acme/CheckoutIT.java",
        "spec/models/user_spec.rb",
        "conftest.py",
        "tests/conftest.py",
        # Windows-style separators still classify.
        "tests\\api\\test_client.py",
    ],
)
def test_recognises_test_and_fixture_files(path: str):
    assert is_test_source(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # Real source and docs.
        "src/onboarding/checks.py",
        "AGENTS.md",
        "docs/process.md",
        "README.md",
        "src/main/java/com/acme/UserService.java",
        "internal/server/server.go",
        # Data/config that merely contains "spec"/"test" — must not false-positive.
        "openapi_spec.yaml",
        "config/manifest.json",
        "docs/testing-guide.md",
        "src/latest.js",
        # Commit artifacts the ingester produces.
        "commit-7766d14af83f8df82de1e7667a629e1914626bbd.md",
        "",
        None,
    ],
)
def test_leaves_real_sources_alone(path: str | None):
    assert is_test_source(path) is False
