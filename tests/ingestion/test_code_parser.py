from ingestion.code_parser import PATTERNS, compute_boundaries, parse_code
from ingestion.parser import parse


def test_parse_python_multiple_functions():
    code = b"""
import os

def foo():
    return 1

def bar():
    return 2
"""

    result = parse("test.py", code)

    assert len(result) >= 2

    contents = " ".join(chunk.content for chunk in result)
    assert "foo" in contents
    assert "bar" in contents


def test_preamble_is_present_in_each_chunk():
    code = b"""
import os
import sys

def foo():
    return 1

def bar():
    return 2
"""

    result = parse("test.py", code)

    for chunk in result:
        assert "import os" in chunk.content
        assert "import sys" in chunk.content

def test_fallback_when_no_definitions():
    code = b"""
print("hello")
x = 1 + 1
"""

    result = parse("test.py", code)

    assert all(chunk.kind == "text" for chunk in result)

def test_fallback_when_file_is_not_supported():
    code = b"""
int add(int a, int b) {
    return a + b;
}
"""

    result = parse_code("test.c", code)

    assert all(chunk.kind == "text" for chunk in result)


def test_class_is_detected():
    code = b"""
import os

class User:
    def __init__(self):
        pass
"""

    result = parse("test.py", code)

    assert any("class User" in chunk.content for chunk in result)

def test_chunk_order_is_preserved():
    code = b"""
import os

def a():
    pass

def b():
    pass
"""

    result = parse("test.py", code)

    indices = [chunk.metadata["chunk_index"] for chunk in result]

    assert indices == sorted(indices)

def test_no_empty_chunks_are_generated():
    code = b"""
def foo():

def bar():
    pass
"""

    result = parse("test.py", code)

    for chunk in result:
        assert chunk.content.strip() != ""

def test_compute_boundaries_detects_functions():
    lines = [
        "import os",
        "def foo():",
        "    pass",
        "def bar():",
        "    pass",
    ]

    pattern = PATTERNS[".py"]

    boundaries = compute_boundaries(lines, pattern)

    assert boundaries == [1, 3]


def test_parse_javascript_arrow_functions_and_classes():
    code = b"""
import React from "react";

const add = (a, b) => a + b;

class User {
    constructor(name) {
        this.name = name;
    }
}

export const multiply = (a, b) => {
    return a * b;
};
"""

    result = parse("test.js", code)

    assert len(result) >= 2
    assert any("add" in chunk.content for chunk in result)
    assert any("User" in chunk.content for chunk in result)
    assert any("multiply" in chunk.content for chunk in result)

def test_parse_typescript_functions_and_classes():
    code = b"""
export const sum = (a: number, b: number) => a + b;

export class User {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}
"""

    result = parse("test.ts", code)

    assert len(result) >= 2
    assert any("sum" in chunk.content for chunk in result)
    assert any("class User" in chunk.content for chunk in result)


def test_parse_go_functions():
    code = b"""
package main

import "fmt"

func add(a int, b int) int {
    return a + b
}

func main() {
    fmt.Println(add(1, 2))
}
"""

    result = parse("test.go", code)

    assert len(result) >= 2
    assert any("add" in chunk.content for chunk in result)
    assert any("main" in chunk.content for chunk in result)