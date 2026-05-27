import pytest

from main import main


def test_main_prints_hello(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "Hello from sprintstart-ai!" in captured.out
