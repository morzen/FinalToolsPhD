"""Display names for the explorer, read from the author's MongoDB dumps (Cyber_-Necronomicon, read only):
technique and software names (latest scrape of each ATT&CK page), technique tactics, CAPEC and CWE names.
Written once to names.json next to this file; nothing else is written."""
import os
import json
import math
from datetime import datetime

from minibson import iter_docs

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'names.json')
VERSION = 2            # 2: tactics of multi-tactic techniques ('Tactics' field) included


def _scrape_time(doc):
    try:
        return datetime.strptime(doc.get('ScrapDate', [''])[0], '%d-%m-%Y_%H-%M-%S')
    except (ValueError, IndexError, TypeError):
        return datetime.min


def _clean(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ''
    return str(s).strip()


def build(dump_dir):
    attack = os.path.join(dump_dir, '1_MITRE_ATTCK_Data.bson')
    capec = os.path.join(dump_dir, '2_MITRE_CAPEC_Data.bson')
    cwe = os.path.join(dump_dir, '3_MITRE_CWE_Data.bson')
    latest = {}
    for d in iter_docs(attack):
        ids = d.get('title', {}).get('ID') or ['']
        i = ids[0]
        if i and (i not in latest or _scrape_time(d) >= _scrape_time(latest[i])):
            latest[i] = d
    out = {'techniques': {}, 'software': {}, 'groups': {}, 'capec': {}, 'cwe': {}, 'version': VERSION,
           'source': 'Cyber_-Necronomicon dumps (ATT&CK pages: latest scrape per ID)'}
    for i, d in latest.items():
        name = _clean((d['title'].get('Name') or [''])[0])
        sut = d.get('SumUpTable', {})
        if i.startswith('T'):
            # single-tactic pages have 'Tactic', multi-tactic pages 'Tactics'
            tactics = [t.strip() for t in ','.join(sut.get('Tactic', []) + sut.get('Tactics', [])).split(',')
                       if t.strip()]
            out['techniques'][i] = {'name': name, 'tactics': tactics}
        elif i.startswith('S'):
            out['software'][i] = {'name': name, 'type': _clean((sut.get('Type') or [''])[0]).lower()}
        elif i.startswith('G'):
            out['groups'][i] = {'name': name, 'aliases': d['title'].get('Associated APT', [])}
    for d in iter_docs(capec):
        cid = _clean(d.get('CAPEC_ID'))
        if cid:
            out['capec'][cid] = {'name': _clean(d.get('Name'))}
    for d in iter_docs(cwe):
        try:
            cid = f"CWE-{int(d.get('CWE-ID'))}"
        except (TypeError, ValueError):
            continue
        out['cwe'][cid] = {'name': _clean(d.get('Name'))}
    return out


def load(dump_dir):
    """names.json if it exists, else build it from the dumps (None if the dumps are not found)."""
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            data = json.load(fh)
        if data.get('version') == VERSION:
            return data
    if not os.path.exists(os.path.join(dump_dir, '1_MITRE_ATTCK_Data.bson')):
        return None
    data = build(dump_dir)
    with open(CACHE, 'w') as fh:
        json.dump(data, fh)
    return data
