import os
import sys


class C:
    enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @classmethod
    def _w(cls, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if cls.enabled else text

    @classmethod
    def dim(cls, t: str) -> str:
        return cls._w("2", t)

    @classmethod
    def bold(cls, t: str) -> str:
        return cls._w("1", t)

    @classmethod
    def cyan(cls, t: str) -> str:
        return cls._w("36", t)

    @classmethod
    def green(cls, t: str) -> str:
        return cls._w("32", t)

    @classmethod
    def yellow(cls, t: str) -> str:
        return cls._w("33", t)

    @classmethod
    def red(cls, t: str) -> str:
        return cls._w("31", t)
