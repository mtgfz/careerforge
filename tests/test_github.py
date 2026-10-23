import pytest

from src.ingestion import loaders
from src.ingestion.loaders import _clean_readme, github_username, load_github


class FakeResponse:
    def __init__(self, status=200, text="", data=None, headers=None):
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


LONG = "Real project description. " * 80  # ~2000 chars
MEDIUM = "Some documentation here. " * 12  # ~300 chars

REPOS = [
    {"name": "empty", "description": None, "language": None, "stargazers_count": 0},
    {"name": "forked", "description": "someone else's", "fork": True, "language": "C", "stargazers_count": 0},
    {"name": "mid", "description": "Medium project", "language": "Python", "stargazers_count": 0},
    {"name": "big", "description": "Big project", "language": "Java", "stargazers_count": 3},
    {"name": "stub", "description": "Only a description", "language": None, "stargazers_count": 0},
]
READMES = {"empty": "hi", "forked": LONG, "mid": MEDIUM, "big": LONG, "stub": ""}


@pytest.fixture
def fake_github(monkeypatch):
    def fake_get(url, **kwargs):
        if url.startswith("https://api.github.com/users/ghost/"):
            return FakeResponse(404)
        if url.startswith("https://api.github.com/"):
            return FakeResponse(data=REPOS)
        repo = url.split("/")[4]
        if url.endswith("/README.md") and READMES.get(repo):
            return FakeResponse(text=READMES[repo])
        return FakeResponse(404)

    monkeypatch.setattr(loaders.requests, "get", fake_get)


def test_clean_readme_strips_html_badges_and_links():
    raw = (
        '<div align="center">\n[![Build](x.svg)](https://ci)\n</div>\n\n---\n\n'
        "## Overview\n\n**Lyrics** turns [your music](http://z) into &amp; text<br/>done\n\n\n\n"
    )
    assert _clean_readme(raw) == "## Overview\n\n**Lyrics** turns your music into & text\ndone"


def test_clean_readme_truncates_on_a_line_boundary():
    out = _clean_readme("line one\nline two\nline three", max_chars=20)
    assert out == "line one\nline two"


def test_empty_repos_and_forks_are_ignored(fake_github):
    sources = [d.source for d in load_github("me")]
    assert "github:me/empty" not in sources
    assert "github:me/forked" not in sources
    assert "github:me/stub" in sources  # no README but has a description


def test_best_repos_are_kept_first(fake_github):
    sources = [d.source for d in load_github("me", max_repos=2)]
    assert sources == ["github:me/big", "github:me/mid"]


def test_forks_can_be_included(fake_github):
    assert "github:me/forked" in [d.source for d in load_github("me", include_forks=True)]


@pytest.mark.parametrize(
    "value",
    [
        "trxshi747",
        " trxshi747 ",
        "@trxshi747",
        "https://github.com/trxshi747",
        "https://github.com/trxshi747/",
        "http://www.github.com/trxshi747/LyricSync?tab=readme",
        "github.com/trxshi747",
    ],
)
def test_github_username_accepts_urls_and_handles(value):
    assert github_username(value) == "trxshi747"


def test_load_github_accepts_a_profile_url(fake_github):
    assert load_github("https://github.com/me/")  # resolved to "me", not "not found"


def test_unknown_user_gives_a_clear_error(fake_github):
    with pytest.raises(ValueError, match="not found"):
        load_github("ghost")
