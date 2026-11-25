"""
tm_data.py -- everything the threat model is built from, read only from the thesis material.

  * The attribution dataset of Section 3.3.5 (FinalToolsPhD_6feat/BaseData/comparison_Encoded_df.csv): the
    730,910 records TTP -> Software -> CAPEC -> CWE -> CVE -> Pulse of the 163 groups.  The profiles are taken
    from the tools themselves (common/protocol.load_profiles) and the attack paths from these records; the two
    are checked to agree.
  * The MongoDB dumps (Cyber_-Necronomicon):
      - 1_MITRE_ATTCK_Data.bson, the ATT&CK pages of the 25 November 2024 scrape (release 16, Section 1.4.2):
        technique names, tactics, platforms, mitigations, detections and procedure examples, software names and
        types;
      - 2_MITRE_CAPEC_Data.bson: CAPEC names, typical severity, likelihood of attack and mitigations;
      - 3_MITRE_CWE_Data.bson: CWE names, potential mitigations (phase and strategy) and detection methods;
      - 0_IDs_level_abstraction_schema.bson, the schema collection: the public exploits recorded per CVE.

Nothing is written anywhere (no bytecode, no cache).
"""
import os
import re
import sys
import math
import collections
from datetime import datetime

sys.dont_write_bytecode = True

import pandas as pd                                                   # noqa: E402

from minibson import iter_docs                                        # noqa: E402

ATTACK_SCRAPE = '25-11-2024'          # the release-16 scrape used by the thesis (Section 1.4.2)
LEVEL_COLUMN = {'techniques': 'techniques_Origin', 'software': 'software_Origin', 'capec': 'CAPEC_ID_Origin',
                'cwe': 'CWE_ID_Origin', 'cve': 'CVE_ID_Origin', 'pulse': 'Pulse_ID_Origin'}
MAP_COLUMN = {'techniques': 'techniques', 'software': 'software', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
              'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}
LEVEL_LABEL = {'techniques': 'TTPs', 'software': 'Software', 'capec': 'CAPEC', 'cwe': 'CWE', 'cve': 'CVE',
               'pulse': 'Pulse'}
# Enterprise tactics of ATT&CK release 16, in kill-chain order; Mobile and ICS tactics follow
ENTERPRISE_TACTICS = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence',
                      'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery',
                      'Lateral Movement', 'Collection', 'Command and Control', 'Exfiltration', 'Impact']


# lifecycle phases used by CWE "Potential Mitigations" (a mitigation can name several, written one after the other)
CWE_PHASES = ['Architecture and Design', 'Build and Compilation', 'Patching and Maintenance', 'System Configuration',
              'Decommissioning and End-of-Life', 'Requirements', 'Implementation', 'Testing', 'Documentation',
              'Bundling', 'Distribution', 'Installation', 'Operation', 'Porting', 'Integration', 'Manufacturing',
              'Policy']


def split_phases(text):
    """'Architecture and Design Operation' -> ['Architecture and Design', 'Operation']."""
    out, rest = [], ' ' + (text or '') + ' '
    for ph in CWE_PHASES:
        if f' {ph} ' in rest:
            out.append(ph)
            rest = rest.replace(f' {ph} ', ' ')
    return out


def _clean(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ''
    return str(s).strip()


def _first(lst, default=''):
    return _clean(lst[0]) if isinstance(lst, list) and lst else default


def _split_colons(text):
    """'::A::B::' -> ['A', 'B'] (the CAPEC/CWE dump format)."""
    return [p.strip() for p in _clean(text).split('::') if p.strip() and p.strip().lower() not in ('na', 'nan')]


def _fields(entry, names):
    """'PHASE:Implementation:STRATEGY:Input Validation:DESCRIPTION:...' -> {'PHASE': ..., 'STRATEGY': ...}."""
    out, key = {}, None
    pattern = r'(' + '|'.join(re.escape(n) for n in names) + r'):'
    parts = re.split(pattern, entry)
    for p in parts:
        if p in names:
            key = p
        elif key:
            out[key] = (out.get(key, '') + p).strip().rstrip(':').strip()
    return out


class ThreatData:
    def __init__(self, tool_root, dump_dir, log=print):
        self.root = os.path.abspath(tool_root)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from common import protocol as P                              # the tools' own protocol (read only)
        self.P = P
        self.features = list(P.FEATURES)
        self.names = P.apt_names()
        self.log = log
        self._load_records()
        self._load_attack(os.path.join(dump_dir, '1_MITRE_ATTCK_Data.bson'))
        self._load_capec(os.path.join(dump_dir, '2_MITRE_CAPEC_Data.bson'))
        self._load_cwe(os.path.join(dump_dir, '3_MITRE_CWE_Data.bson'))
        self._load_exploits(os.path.join(dump_dir, '0_IDs_level_abstraction_schema.bson'))

    # ------------------------------------------------------------------ dataset records and profiles
    def _load_records(self):
        P = self.P
        cols = ['APTnameOrigin'] + [LEVEL_COLUMN[f] for f in self.features]
        enc_cols = [f"{MAP_COLUMN[f]}_Encoded" for f in self.features]
        df = pd.read_csv(os.path.join(P.BASE, 'comparison_Encoded_df.csv'), usecols=cols + enc_cols,
                         dtype=str, keep_default_na=True)
        self.n_records = len(df)
        # origin <-> encoded per level (the dataset mapping; one-to-one)
        self.to_origin = {}
        for f in self.features:
            pair = df[[LEVEL_COLUMN[f], f"{MAP_COLUMN[f]}_Encoded"]].dropna()
            pair = pair[pair[f"{MAP_COLUMN[f]}_Encoded"].astype(int) != P.MISSING]
            self.to_origin[f] = dict(zip(pair[f"{MAP_COLUMN[f]}_Encoded"].astype(int), pair[LEVEL_COLUMN[f]]))
        # profiles: the tools' own profiles, in origin IDs
        tool_profiles = P.load_profiles()
        self.profiles = {g: {f: frozenset(self.to_origin[f][e] for e in tool_profiles[g][f]) for f in self.features}
                         for g in self.names}
        # the same profiles rebuilt from the records (check)
        rebuilt = {g: {f: set() for f in self.features} for g in self.names}
        df = df[cols]
        df.columns = ['apt'] + self.features
        for f in self.features:
            for g, vals in df[['apt', f]].dropna().groupby('apt')[f]:
                rebuilt[g][f] = set(vals)
        bad = [(g, f) for g in self.names for f in self.features if rebuilt[g][f] != set(self.profiles[g][f])]
        if bad:
            raise RuntimeError(f"profiles differ from the records for {len(bad)} group/level pairs, e.g. {bad[:3]}")
        # attack paths: distinct (TTP, CAPEC, CWE, CVE) chains of each group (records where all four are present)
        chain = df.dropna(subset=['techniques', 'capec', 'cwe', 'cve'])
        self.paths = {g: set() for g in self.names}
        for g, t, c, w, v in chain[['apt', 'techniques', 'capec', 'cwe', 'cve']].drop_duplicates().itertuples(index=False):
            self.paths[g].add((t, c, w, v))
        # the software a group uses with each technique (records where both are present)
        pairs = df[['apt', 'techniques', 'software']].dropna().drop_duplicates()
        self.tech_soft = {g: {} for g in self.names}
        for g, t, sw in pairs.itertuples(index=False):
            self.tech_soft[g].setdefault(t, set()).add(sw)
        # links between consecutive levels anywhere in the dataset
        self.links = {}
        for a, b in (('techniques', 'capec'), ('capec', 'cwe'), ('cwe', 'cve'), ('cve', 'pulse'), ('techniques', 'software')):
            sub = df[[a, b]].dropna().drop_duplicates()
            self.links[(a, b)] = set(map(tuple, sub.itertuples(index=False)))
        self.cve_pulses = {}
        for v, pl in self.links[('cve', 'pulse')]:
            self.cve_pulses.setdefault(v, set()).add(pl)
        self.log(f"[tm] {self.n_records:,} records, {len(self.names)} groups; profiles identical to the tools' profiles")

    # ------------------------------------------------------------------ ATT&CK release 16 (25 Nov 2024 scrape)
    def _load_attack(self, path):
        latest_any = {}
        self.tech, self.soft, self.groups = {}, {}, {}
        scraped = []
        for d in iter_docs(path):
            i = _first(d.get('title', {}).get('ID'))
            if not i:
                continue
            when = _first(d.get('ScrapDate'))
            if when.startswith(ATTACK_SCRAPE):
                scraped.append(d)
            else:
                t = datetime.strptime(when, '%d-%m-%Y_%H-%M-%S') if when else datetime.min
                if i not in latest_any or t > latest_any[i][0]:
                    latest_any[i] = (t, d)
        for d in scraped:
            self._add_attack_doc(d, release16=True)
        # techniques of the dataset that release 16 no longer has (deprecated): name from the last scrape that has it
        for i, (_, d) in latest_any.items():
            if i.startswith('T') and i not in self.tech:
                self._add_attack_doc(d, release16=False)

    def _add_attack_doc(self, d, release16):
        i = _first(d['title'].get('ID'))
        name = _first(d['title'].get('Name'))
        sut = d.get('SumUpTable', {}) or {}
        tables = d.get('HeadersAndTables') or {}
        if i.startswith('T'):
            tactic_text = ','.join((sut.get('Tactic') or []) + (sut.get('Tactics') or []))
            tactics = [t.strip() for t in tactic_text.split(',') if t.strip()]
            mitig = {}
            mt = tables.get('Mitigations')
            for mid, rows in (mt.items() if isinstance(mt, dict) else []):
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    mitig[mid] = _clean(rows[0].get('Mitigation'))
            detect = {}
            dt = tables.get('Detection')
            for did, rows in (dt.items() if isinstance(dt, dict) else []):
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict):
                            src, comp = _clean(r.get('Data Source')), _clean(r.get('Data Component'))
                            if src:
                                detect[f"{did}:{comp or src}"] = f"{src}: {comp}" if comp else src
            procedures = {}
            pe = tables.get('Procedure Examples')
            for pid, rows in (pe.items() if isinstance(pe, dict) else []):
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    text = re.sub(r'\s*https?://\S+', '', _clean(rows[0].get('Description'))).strip()
                    procedures[pid] = text
            self.tech[i] = {'name': name, 'tactics': tactics, 'domain': _first(d['title'].get('Domain')),
                            'platforms': [p.strip() for p in ','.join(sut.get('Platforms') or []).split(',') if p.strip()],
                            'mitigations': mitig, 'detections': detect, 'procedures': procedures,
                            'release16': release16}
        elif i.startswith('S'):
            self.soft[i] = {'name': name, 'type': _first(sut.get('Type')).lower()}
        elif i.startswith('G'):
            self.groups[i] = {'name': name, 'aliases': d['title'].get('Associated APT', [])}

    # ------------------------------------------------------------------ CAPEC 3.9, CWE 4.17, schema exploits
    def _load_capec(self, path):
        self.capec = {}
        for d in iter_docs(path):
            cid = _clean(d.get('CAPEC_ID'))
            if not cid:
                continue
            self.capec[cid] = {'name': _clean(d.get('Name')),
                               'severity': _clean(d.get('Typical Severity')) or 'not stated',
                               'likelihood': _clean(d.get('Likelihood Of Attack')) or 'not stated',
                               'mitigations': _split_colons(d.get('Mitigations'))}

    def _load_cwe(self, path):
        self.cwe = {}
        for d in iter_docs(path):
            cid = _clean(d.get('CWE_ID'))
            if not cid:
                try:
                    cid = f"CWE-{int(d.get('CWE-ID'))}"
                except (TypeError, ValueError):
                    continue
            mit = []
            for e in _split_colons(d.get('Potential Mitigations')):
                f = _fields(e, ['PHASE', 'STRATEGY', 'DESCRIPTION', 'EFFECTIVENESS', 'EFFECTIVENESS_NOTES'])
                mit.append((f.get('PHASE', ''), f.get('STRATEGY', '')))
            det = []
            for e in _split_colons(d.get('Detection Methods')):
                f = _fields(e, ['METHOD', 'DESCRIPTION', 'EFFECTIVENESS', 'EFFECTIVENESS_NOTES'])
                if f.get('METHOD'):
                    det.append(f['METHOD'])
            self.cwe[cid] = {'name': _clean(d.get('Name')), 'mitigations': mit, 'detection': det}

    def _load_exploits(self, path):
        """Number of public exploits the schema collection records for each CVE (Section 3.3.4, Figure 3.7)."""
        self.exploits = collections.Counter()
        seen = set()
        for d in iter_docs(path):
            for t, capecs in d.items():
                if t == '_id' or not isinstance(capecs, dict):
                    continue
                for cwes in capecs.values():
                    if not isinstance(cwes, dict):
                        continue
                    for cves in cwes.values():
                        if not isinstance(cves, dict):
                            continue
                        for cve, info in cves.items():
                            if cve in seen or not isinstance(info, dict):
                                continue
                            seen.add(cve)
                            self.exploits[cve] = len(info.get('exploits') or [])

    # ------------------------------------------------------------------ helpers
    def label(self, level, x):
        if level == 'techniques':
            return self.tech.get(x, {}).get('name', '')
        if level == 'software':
            return self.soft.get(x, {}).get('name', '')
        if level == 'capec':
            return self.capec.get(x, {}).get('name', '')
        if level == 'cwe':
            return self.cwe.get(x, {}).get('name', '')
        return ''

    def tactics(self, t):
        return self.tech.get(t, {}).get('tactics', [])
