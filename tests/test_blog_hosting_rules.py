"""Offline hosting-contract tests. A Netlify preview is still required.

Model the intended file-shadowing/ordered rewrite contract explicitly. This is
not Netlify's runtime and does not claim to replace a deployed HTTP check.
"""
from pathlib import Path
import fnmatch
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'frontend'


def config_rules(text):
    # Restricted parser for this repository's three-field redirects only.
    result = []
    for block in text.split('[[redirects]]')[1:]:
        fields = dict(re.findall(r'^(from|to|status) = (.+)$', block, re.M))
        if set(fields) != {'from', 'to', 'status'}:
            raise ValueError('Unexpected redirect fields')
        result.append({'from': fields['from'].strip('"'),
                       'to': fields['to'].strip('"'),
                       'status': int(fields['status'])})
    return result


def backup_rules(text):
    return [dict(zip(('from', 'to', 'status'), [a, b, int(c)]))
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith('#')
            for a, b, c in [line.split()]]


def resolve(path, rules, dist):
    # Existing static files shadow non-forced rewrites on Netlify.
    f = dist / path.lstrip('/')
    if f.is_file() or (f / 'index.html').is_file():
        return 200, path
    for rule in rules:
        if fnmatch.fnmatchcase(path, rule['from']):
            return rule['status'], rule['to']
    return 404, None


class BlogHostingRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dist = ROOT / 'dist'
        if not (cls.dist / 'sitemap.xml').is_file():
            raise RuntimeError('Run the full prerender build first')
        cls.rule_sets = [config_rules((ROOT / 'netlify.toml').read_text()),
                         backup_rules((ROOT / 'public/_redirects').read_text())]
        cls.urls = [e.text for e in ET.parse(cls.dist / 'sitemap.xml').getroot()
                    .findall('{*}url/{*}loc')]

    def test_sources_agree(self):
        def compact(rules): return [(r['from'], r['to'], r['status']) for r in rules]
        self.assertEqual(compact(self.rule_sets[0]), compact(self.rule_sets[1]))

    def test_all_sitemap_paths_have_static_indexes(self):
        from urllib.parse import urlparse
        self.assertEqual(len(self.urls), 13)
        for url in self.urls:
            path = urlparse(url).path
            html = (self.dist / path.lstrip('/') / 'index.html').read_text()
            self.assertIn('rel="canonical" href="' + url + '"', html)
            for rules in self.rule_sets:
                self.assertEqual(resolve(path, rules, self.dist), (200, path))

    def test_unknown_blog_paths_reach_real404(self):
        for path in ['/blog/not-a-real-article-probe/', '/blog/not-real',
                     '/blog/not-real/child', '/blog/no-such-diagram.svg']:
            for rules in self.rule_sets:
                self.assertEqual(resolve(path, rules, self.dist), (404, '/404.html'))

    def test_existing_app_paths_unchanged(self):
        for path in ['/my-tests', '/auth', '/practice', '/view-test/test-id',
                     '/take-test/test-id', '/submit-feedback/test-id']:
            for rules in self.rule_sets:
                self.assertEqual(resolve(path, rules, self.dist), (200, '/index.html'))

    def test_old_wildcard_reproduces_defect(self):
        for rules in self.rule_sets:
            broken = [dict(r) for r in rules]
            broken.insert(-1, {'from': '/blog/*', 'to': '/index.html', 'status': 200})
            self.assertEqual(resolve('/blog/not-a-real-article-probe/', broken, self.dist),
                             (200, '/index.html'))

    def test_no_catchall_blog_app_rule(self):
        for rules in self.rule_sets:
            self.assertFalse(any(r['from'] == '/blog/*' for r in rules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
