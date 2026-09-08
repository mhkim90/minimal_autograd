"""Pytest adapter for the loop's script-style temporary-directory suites."""
import pytest


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)
