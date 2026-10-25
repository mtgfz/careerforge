"""Import a job offer from its URL.

Most job boards embed a schema.org `JobPosting` in JSON-LD (it is what Google Jobs reads), so
that is tried first; otherwise the readable text of the page is extracted from the HTML.
Only public http(s) addresses are fetched: private, loopback and link-local hosts are refused
so the API cannot be used to probe the internal network.
"""
from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5
MIN_TEXT_CHARS = 200  # below this the page probably needs JavaScript or a login
USER_AGENT = "Mozilla/5.0 (compatible; CareerForge/0.1; +https://github.com/mtgfz/careerforge)"


class JobImportError(ValueError):
    """The URL could not be turned into a job offer text."""


@dataclass
class ImportedJob:
    url: str
    text: str
    title: str = ""
    company: str = ""
    method: str = ""  # "json-ld" or "html"


# ---------- URL safety ----------
def _check_public(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise JobImportError("Only http(s) URLs are supported")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise JobImportError(f"Cannot resolve host '{parts.hostname}'") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise JobImportError(f"Refusing to fetch a non-public address ({parts.hostname})")


def _download(url: str, timeout: float = 15) -> tuple[str, str]:
    """GET with manual redirects (each hop is checked) and a size cap. Returns (final_url, html)."""
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as e:
            raise JobImportError(f"Could not reach {url}: {e}") from e
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise JobImportError("Redirect without a Location header")
            url = urljoin(url, location)
            continue
        if resp.status_code in (401, 403, 429, 999):
            raise JobImportError(
                f"The site refused the request (HTTP {resp.status_code}); "
                "copy and paste the offer text instead"
            )
        if not resp.ok:
            raise JobImportError(f"HTTP {resp.status_code} for {url}")
        ctype = resp.headers.get("Content-Type", "")
        if ctype and "html" not in ctype and "text" not in ctype:
            raise JobImportError(f"Not a web page ({ctype})")
        body = b""
        for chunk in resp.iter_content(64 * 1024):
            body += chunk
            if len(body) > MAX_BYTES:
                break
        resp.close()
        return url, body[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace")
    raise JobImportError("Too many redirects")


# ---------- JSON-LD JobPosting ----------
_LD_BLOCK = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.S | re.I
)


def _iter_ld(node):
    if isinstance(node, list):
        for n in node:
            yield from _iter_ld(n)
    elif isinstance(node, dict):
        yield node
        yield from _iter_ld(node.get("@graph", []))


def _is_posting(node: dict) -> bool:
    kind = node.get("@type")
    kinds = kind if isinstance(kind, list) else [kind]
    return "JobPosting" in kinds


def _name(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    if isinstance(value, list):
        return ", ".join(filter(None, (_name(v) for v in value)))
    return str(value or "")


def _location(value) -> str:
    places = value if isinstance(value, list) else [value]
    out = []
    for p in places:
        addr = p.get("address", p) if isinstance(p, dict) else p
        if isinstance(addr, dict):
            out.append(", ".join(
                str(addr[k]) for k in ("addressLocality", "addressRegion", "addressCountry")
                if isinstance(addr.get(k), str) and addr.get(k)
            ))
        elif addr:
            out.append(str(addr))
    return "; ".join(filter(None, out))


def _from_json_ld(page: str) -> ImportedJob | None:
    for raw in _LD_BLOCK.findall(page):
        try:
            data = json.loads(html.unescape(raw.strip()))
        except json.JSONDecodeError:
            continue
        for node in _iter_ld(data):
            if not _is_posting(node):
                continue
            title = _name(node.get("title"))
            company = _name(node.get("hiringOrganization"))
            lines = [f"{title} - {company}" if company else title]
            loc = _location(node.get("jobLocation"))
            if loc:
                lines.append(f"Location: {loc}")
            if node.get("employmentType"):
                lines.append(f"Employment type: {_name(node['employmentType'])}")
            lines.append("")
            lines.append(html_to_text(str(node.get("description") or "")))
            for key, label in (
                ("responsibilities", "Responsibilities"),
                ("qualifications", "Qualifications"),
                ("skills", "Skills"),
                ("experienceRequirements", "Experience"),
                ("educationRequirements", "Education"),
            ):
                val = node.get(key)
                if val:
                    val = _name(val) if not isinstance(val, str) else html_to_text(val)
                    lines.append(f"\n{label}:\n{val}")
            text = _tidy("\n".join(lines))
            if len(text) >= 50:
                return ImportedJob(url="", text=text, title=title, company=company, method="json-ld")
    return None


# ---------- plain HTML ----------
_SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form",
         "button", "iframe", "template", "head"}
_BLOCK = {"p", "div", "section", "article", "main", "br", "li", "ul", "ol", "tr", "table",
          "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt", "blockquote", "pre"}
_VOID = {"br", "img", "hr", "input", "meta", "link", "source", "wbr", "area", "col", "embed"}
# A class/id token naming page furniture: "related-posts", "jeg_ad", "offres-emploi-similaire"...
_NOISE_TOKEN = re.compile(
    r"^(?:.*[-_])?(?:related|similar|similaire|sidebar|widgets?|footer|comments?|share|sharing|"
    r"social|ads?|advert\w*|popup|modal|menu|breadcrumbs?|newsletter|cookies?|signup|subscribe|"
    r"promo)(?:[-_].*)?$",
    re.I,
)


def _is_noise(attrs) -> bool:
    for name, value in attrs:
        if name in ("class", "id") and value:
            for token in value.split():
                if not token.lower().startswith("no-") and _NOISE_TOKEN.match(token):
                    return True
    return False


class _TextExtractor(HTMLParser):
    """Collects the page text without page furniture, plus each <main>/<article> block on its
    own and the position of the first <h1> (where the offer usually starts)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []  # whole page
        self.blocks: list[dict] = []  # one per <main>/<article>: {"parts": [...], "h1": bool}
        self._open: list[dict] = []
        self._skip: list[list] = []  # [tag, nesting] of the element being skipped
        self.h1_at: int | None = None
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if self._skip:
            if tag == self._skip[-1][0] and tag not in _VOID:
                self._skip[-1][1] += 1
            return
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return
        # A <header> inside the content (WordPress "entry-header") holds the title: keep it.
        page_chrome = tag in _SKIP and not (tag == "header" and self._open)
        if page_chrome or _is_noise(attrs):
            self._skip.append([tag, 1])
            return
        if tag in ("main", "article"):
            block = {"parts": [], "h1": False}
            self.blocks.append(block)
            self._open.append(block)
        if tag == "h1" and self.h1_at is None:
            self.h1_at = len(self.parts)
            for block in self._open:
                block["h1"] = True
        if tag == "li":
            self._emit("\n- ")
        elif tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if self._skip:
            if tag == self._skip[-1][0]:
                self._skip[-1][1] -= 1
                if not self._skip[-1][1]:
                    self._skip.pop()
            return
        if tag in _VOID:
            return
        if tag in _BLOCK:
            self._emit("\n")
        if tag in ("main", "article") and self._open:
            self._open.pop()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self._skip:
            self._emit(data)

    def _emit(self, s):
        self.parts.append(s)
        for block in self._open:
            block["parts"].append(s)


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln != "-")  # drop empty list items
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(fragment: str) -> str:
    p = _TextExtractor()
    p.feed(fragment)
    p.close()
    return _tidy("".join(p.parts))


def _from_html(page: str) -> ImportedJob:
    """Pick the offer: the <main>/<article> holding the <h1>, else the page from its <h1> on,
    else the longest <main>/<article>, else the whole page (furniture removed in every case)."""
    p = _TextExtractor()
    p.feed(page)
    p.close()
    blocks = [(_tidy("".join(b["parts"])), b["h1"]) for b in p.blocks]
    candidates = [t for t, has_h1 in blocks if has_h1]
    if p.h1_at is not None:
        candidates.append(_tidy("".join(p.parts[p.h1_at:])))
    candidates.append(max((t for t, _ in blocks), key=len, default=""))
    text = next((t for t in candidates if len(t) >= MIN_TEXT_CHARS), _tidy("".join(p.parts)))
    return ImportedJob(url="", text=text, title=_tidy(p.title), method="html")


# ---------- public API ----------
def parse_job_page(page: str) -> ImportedJob:
    """Turn a job page's HTML into an ImportedJob (no network access)."""
    return _from_json_ld(page) or _from_html(page)


def import_job_from_url(url: str, timeout: float = 15) -> ImportedJob:
    url = url.strip()
    if not urlparse(url).scheme:
        url = "https://" + url
    final_url, page = _download(url, timeout=timeout)
    job = parse_job_page(page)
    job.url = final_url
    if len(job.text) < MIN_TEXT_CHARS:
        raise JobImportError(
            "Too little text found on the page (it may need JavaScript or a login); "
            "copy and paste the offer text instead"
        )
    return job


if __name__ == "__main__":  # python -m src.ingestion.job_url <url> > job.txt
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.ingestion.job_url <url>")
    try:
        print(import_job_from_url(sys.argv[1]).text)
    except JobImportError as e:
        sys.exit(f"error: {e}")
