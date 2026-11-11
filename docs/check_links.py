#!/usr/bin/env python3
"""Check local pages, fragments and assets in a built DocumenterVitepress site."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/build/1').resolve()
if not (root / 'index.html').is_file():
    raise SystemExit(f'No built site at {root}')
base = '/NLMEEstimation.jl/'
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if attrs.get(key): self.links.append((tag, attrs[key]))

documents = {}
for path in root.rglob('*.html'):
    page = Links(); page.feed(path.read_text()); documents[path] = page
errors = []; checked = 0
for page, parsed in documents.items():
    for tag, link in parsed.links:
        url = urlsplit(link)
        if url.scheme or url.netloc: continue
        path = unquote(url.path)
        if path.startswith(base): target = root / path[len(base):]
        elif path.startswith('/'): target = root / path.lstrip('/')
        else: target = page.parent / path if path else page
        target = target.resolve()
        candidates = [target, target / 'index.html']
        if not target.suffix: candidates.append(target.with_suffix('.html'))
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            errors.append(f'{page.relative_to(root)}: missing {link}')
        elif url.fragment and found in documents and unquote(url.fragment) not in documents[found].ids:
            errors.append(f'{page.relative_to(root)}: missing fragment {link}')
        checked += 1
for error in sorted(set(errors)): print(error)
print(f'Checked {checked} local links in {len(documents)} HTML files; {len(set(errors))} errors')
raise SystemExit(bool(errors))
