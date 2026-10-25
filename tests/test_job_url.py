"""Job import by URL: parsing (JSON-LD and plain HTML) and the network guard rails, offline."""
import json
import socket

import pytest

from src.ingestion import job_url
from src.ingestion.job_url import JobImportError, import_job_from_url, parse_job_page

POSTING = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "Data Engineer Intern",
    "hiringOrganization": {"@type": "Organization", "name": "Example Bank"},
    "jobLocation": {"@type": "Place", "address": {"addressLocality": "Casablanca", "addressCountry": "MA"}},
    "employmentType": "INTERN",
    "description": "<p>Build <b>data pipelines</b>.</p><ul><li>Python &amp; SQL</li><li>Airflow</li></ul>",
    "skills": ["Spark", "Docker"],
}

FILLER = "We build reliable data products for millions of customers. " * 6


def page_with_ld(node) -> str:
    return (
        "<html><head><title>Jobs</title>"
        f'<script type="application/ld+json">{json.dumps(node)}</script></head>'
        "<body><nav>Home | Careers</nav><p>Apply now</p></body></html>"
    )


def test_json_ld_job_posting_is_preferred():
    job = parse_job_page(page_with_ld(POSTING))
    assert job.method == "json-ld"
    assert job.title == "Data Engineer Intern" and job.company == "Example Bank"
    assert job.text.startswith("Data Engineer Intern - Example Bank")
    assert "Casablanca, MA" in job.text
    assert "- Python & SQL" in job.text and "- Airflow" in job.text
    assert "Spark, Docker" in job.text
    assert "<b>" not in job.text and "Careers" not in job.text


def test_json_ld_inside_a_graph_with_type_list():
    node = {"@graph": [{"@type": "WebPage"}, {**POSTING, "@type": ["JobPosting"]}]}
    assert parse_job_page(page_with_ld([node])).method == "json-ld"


def test_html_fallback_keeps_main_content_and_drops_chrome():
    page = f"""<html><head><title>ML Intern | Example</title><style>p{{color:red}}</style></head>
    <body><header>Logo Menu</header><nav>Jobs Blog</nav>
    <main><h1>ML Intern</h1><p>{FILLER}</p><ul><li>PyTorch</li><li>FastAPI</li></ul></main>
    <script>track()</script><footer>Cookies</footer></body></html>"""
    job = parse_job_page(page)
    assert job.method == "html" and job.title == "ML Intern | Example"
    assert "ML Intern" in job.text and "- PyTorch" in job.text
    for noise in ("Logo", "Jobs Blog", "track()", "Cookies", "color:red"):
        assert noise not in job.text


class FakeResponse:
    def __init__(self, status=200, body="", headers=None):
        self.status_code = status
        self.headers = {"Content-Type": "text/html; charset=utf-8", **(headers or {})}
        self._body = body.encode()
        self.encoding = "utf-8"

    ok = property(lambda self: self.status_code < 400)
    is_redirect = property(lambda self: self.status_code in (301, 302, 303, 307, 308))

    def iter_content(self, size):
        yield self._body

    def close(self):
        pass


@pytest.fixture
def net(monkeypatch):
    """Fake DNS (*.internal -> 10.0.0.5, localhost -> 127.0.0.1, rest public) and HTTP."""
    pages: dict[str, FakeResponse] = {}
    calls: list[str] = []

    def getaddrinfo(host, port, *a, **kw):
        ip = {"localhost": "127.0.0.1"}.get(host, "10.0.0.5" if host.endswith(".internal") else "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 443))]

    def get(url, **kw):
        assert kw.get("allow_redirects") is False  # every hop must be re-checked
        calls.append(url)
        return pages[url]

    monkeypatch.setattr(job_url.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(job_url.requests, "get", get)
    return pages, calls


def test_import_follows_redirects_and_sets_the_final_url(net):
    pages, calls = net
    pages["https://jobs.example.com/1"] = FakeResponse(302, headers={"Location": "/offer/1"})
    pages["https://jobs.example.com/offer/1"] = FakeResponse(200, page_with_ld({**POSTING, "description": FILLER}))
    job = import_job_from_url("jobs.example.com/1")  # scheme added
    assert job.url == "https://jobs.example.com/offer/1"
    assert calls == ["https://jobs.example.com/1", "https://jobs.example.com/offer/1"]


@pytest.mark.parametrize("url", ["http://localhost:8000/admin", "https://db.internal/x", "ftp://example.com/a"])
def test_refuses_private_hosts_and_other_schemes(net, url):
    with pytest.raises(JobImportError):
        import_job_from_url(url)
    assert net[1] == []  # nothing was fetched


def test_refuses_a_redirect_to_a_private_host(net):
    pages, calls = net
    pages["https://jobs.example.com/r"] = FakeResponse(302, headers={"Location": "http://169.254.169.254.internal/"})
    with pytest.raises(JobImportError, match="non-public"):
        import_job_from_url("https://jobs.example.com/r")
    assert calls == ["https://jobs.example.com/r"]


def test_blocked_or_empty_pages_give_a_readable_error(net):
    pages, _ = net
    pages["https://www.linkedin.com/jobs/1"] = FakeResponse(999)
    pages["https://spa.example.com/"] = FakeResponse(200, "<html><body><div id=root></div></body></html>")
    with pytest.raises(JobImportError, match="paste the offer text"):
        import_job_from_url("https://www.linkedin.com/jobs/1")
    with pytest.raises(JobImportError, match="JavaScript"):
        import_job_from_url("https://spa.example.com/")
