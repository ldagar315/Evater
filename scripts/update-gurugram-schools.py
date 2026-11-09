#!/usr/bin/env python3
"""Refresh a partial school suggestion directory. Never runs in the client.
Requires requests + beautifulsoup4. Review the diff and source counts before commit.
No contact details, principal names, credentials or student data are retained.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE = 'https://gurugram.gov.in/public-utility-category/schools/'
SARAS = 'https://saras.cbse.gov.in/saras/AffiliatedList/ListOfSchdirReport'
ROOT = Path(__file__).resolve().parents[1]

def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower().replace('gurgaon', 'gurugram')).strip()

def cells_segment(cell, start, end):
    text = cell.get_text(' ', strip=True)
    return text.split(start, 1)[1].split(end, 1)[0].strip()

def cbse(html):
    entries = []
    for row in BeautifulSoup(html, 'html.parser').select('table tbody tr'):
        cols = row.find_all('td')
        if len(cols) < 7:
            continue
        if 'GURUGRAM' not in cols[2].get_text(' ', strip=True).upper():
            raise ValueError('Unexpected district in SARAS response')
        affiliation = re.search(r'Aff\. No\.\s*:\s*(\d+)', cols[1].get_text(' ', strip=True)).group(1)
        name = cells_segment(cols[4], 'Name :', 'Head/Principal Name:')
        address = cells_segment(cols[5], 'Address :', 'Website :')
        entries.append(dict(id=f'cbse-{affiliation}', name=name, address=address, label=f'{name} - {address}', source=urljoin(SARAS, cols[6].find('a')['href'])))
    if not entries:
        raise ValueError('No CBSE entries parsed: source may have changed')
    return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cbse-html', type=Path, help='Use a saved source response instead of fetching SARAS')
    parser.add_argument('--district-json', type=Path, help='Use saved list of {name,source} records instead of fetching district pages')
    parser.add_argument('--output', type=Path, default=ROOT/'frontend/src/data/gurugramSchools.json')
    args = parser.parse_args()
    session = requests.Session()
    if args.cbse_html:
        saras_html = args.cbse_html.read_text()
    else:
        form = session.get(SARAS, timeout=30); form.raise_for_status()
        token = BeautifulSoup(form.text, 'html.parser').select_one('input[name=__RequestVerificationToken]')['value']
        result = session.post(SARAS, data={'__RequestVerificationToken':token, 'MainRadioValue':'State_wise', 'State':'5', 'District':'05'}, timeout=30)
        result.raise_for_status(); saras_html = result.text
    entries = cbse(saras_html)
    if args.district_json:
        district = json.loads(args.district_json.read_text())
    else:
        district = []
        for page in range(1, 101):
            url = BASE if page == 1 else BASE + f'page/{page}/'
            response = session.get(url, timeout=30)
            if response.status_code == 404:
                break
            response.raise_for_status()
            names = [heading.get_text(' ', strip=True) for heading in BeautifulSoup(response.text, 'html.parser').select('h2.heading4')]
            if not names:
                break
            district.extend(dict(name=name, source=url) for name in names)
            time.sleep(.5)
        else:
            raise ValueError('Pagination limit reached; inspect source before saving')
    seen = {normalized(item['name']) for item in entries}
    for item in district:
        key = normalized(item['name'])
        if not key or key in seen:
            continue
        seen.add(key)
        entries.append(dict(id='district-'+hashlib.sha256(key.encode()).hexdigest()[:12], name=item['name'], address='Gurugram district', label=item['name'], source=item['source']))
    entries.sort(key=lambda item: item['label'].lower())
    if len({item['id'] for item in entries}) != len(entries):
        raise ValueError('Duplicate identifiers')
    args.output.write_text(json.dumps(entries, ensure_ascii=False, indent=2)+'\n')
    print(f'Saved {len(entries)} suggestions to {args.output}. Review source coverage and refresh the source date in SchoolSearch.tsx and docs/gurugram-school-directory.md.')

if __name__ == '__main__':
    main()
