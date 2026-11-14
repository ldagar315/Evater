#!/usr/bin/env python3
"""Guard explicitly allowlisted app routes in both Netlify rule sources."""
from pathlib import Path
import re
root = Path(__file__).resolve().parent
routes = set(re.findall(r'<Route path="([^":*]+)"', (root/'src/App.tsx').read_text()))
# Public/prerendered destinations resolve via existing files or explicit rules.
routes -= {'/', '/blog'}
config = (root/'netlify.toml').read_text()
backup = (root/'public/_redirects').read_text()
for route in sorted(routes):
    rule = rf'from = "{re.escape(route)}"\s+to = "/index.html"\s+status = 200'
    assert re.search(rule, config), f'Missing Netlify app fallback: {route}'
    assert f'{route} /index.html 200' in backup.splitlines(), f'Missing backup fallback: {route}'
for text in (config, backup):
    assert text.index('/my-tests') < text.rindex('/404.html'), 'Known routes must precede real404 fallback'
assert 'status = 404' in config and '/* /404.html 404' in backup
# A blog wildcard shadows the real-404 rule and serves home HTML to unknown slugs.
assert not re.search(r'from = "/blog/\*"', config), 'Blog wildcard hides unknown-page404 in netlify.toml'
assert not any(line.strip().startswith('/blog/* ') for line in backup.splitlines()), 'Blog wildcard hides unknown-page404 in _redirects'
print(f'PASS: {len(routes)} static app paths including /my-tests have200 app fallbacks in both rule sources; unknown paths retain404.')
