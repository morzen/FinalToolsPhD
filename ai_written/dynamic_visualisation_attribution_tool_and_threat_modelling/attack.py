"""
attack.py -- the attack generator of the explorer.

For one group, it builds a possible attack from the group's profile and the data-level abstraction of the dataset:
one technique of the group for each ATT&CK tactic it covers, in kill-chain order (release 16), and for each step
the software the group uses with that technique, how ATT&CK says the group (or its software) did it, one attack
path of the group's records TTP -> CAPEC -> CWE -> CVE -> Pulse, and the ATT&CK mitigations and detections.

The technique of a tactic is drawn at random among the group's techniques for that tactic, each equally likely
(the dataset records which techniques a group uses, not how often), and the draw is seeded, so a seed always gives
the same attack.

The user can also choose which tactics get a step drawn from the profile, and give items that the attack must
contain, at any level (techniques, software, CAPEC, CWE, CVE, pulses), typed or taken from a CTI report:
  - an included technique is a step of its own, under its tactic;
  - included software is attached to a step whose technique it implements (ATT&CK procedure examples or the
    dataset records), or gets a step of its own;
  - an included CAPEC, CWE, CVE or pulse becomes the attack path of a step whose technique leads to it in the
    group's records (or, if the group has none, in another group's records).
The ticked tactics that no included item covers are then filled from the group's profile as above.  Without any
included item and with the default tactics, the attack is exactly the one of the plain generator.

Everything shown comes from the dataset and the MongoDB dumps (read with tm_data.py); nothing is invented.
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# Enterprise tactics in kill-chain order, with the Mobile and ICS ones at their place
TACTIC_ORDER = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence',
                'Privilege Escalation', 'Defense Evasion', 'Evasion', 'Credential Access', 'Discovery',
                'Lateral Movement', 'Collection', 'Command and Control', 'Exfiltration', 'Inhibit Response Function',
                'Impair Process Control', 'Impact']
OBJECTIVE_DROP = {'espionage': {'Impact', 'Inhibit Response Function', 'Impair Process Control'},
                  'destruction': {'Exfiltration'}}
INCLUDE_LEVELS = ['techniques', 'software', 'capec', 'cwe', 'cve', 'pulse']
PATH_INDEX = {'capec': 1, 'cwe': 2, 'cve': 3}
PRE_PLATFORM = 'PRE'
PRE_COMPROMISE = 'M1056'


class AttackGenerator:
    def __init__(self, tool_root, dump_dir, log=print):
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        from tm_data import ThreatData                                 # noqa: E402
        self.data = ThreatData(tool_root, dump_dir, log=log)
        d = self.data
        self.gid = {v['name']: k for k, v in d.groups.items()}
        self.aliases = {v['name']: v.get('aliases', []) for v in d.groups.values()}
        # chains (TTP, CAPEC, CWE, CVE) of each group, by technique
        self.chains = {}
        for g in d.names:
            by_t = {}
            for (t, ca, w, v) in d.paths[g]:
                by_t.setdefault(t, []).append((ca, w, v))
            self.chains[g] = {t: sorted(lst) for t, lst in by_t.items()}
        # the chains of all groups together (for included items that the chosen group's records do not reach)
        self.all_paths = sorted(set().union(*d.paths.values())) if d.paths else []
        # techniques each software implements: ATT&CK procedure examples (release 16) and the dataset records
        self.soft_techs = {}
        for t, info in d.tech.items():
            for actor in info.get('procedures', {}):
                if actor.startswith('S'):
                    self.soft_techs.setdefault(actor, set()).add(t)
        for t, sw in d.links.get(('techniques', 'software'), ()):
            self.soft_techs.setdefault(sw, set()).add(t)

    # ------------------------------------------------------------------ helpers
    def _item(self, level, x):
        return {'id': x, 'name': self.data.label(level, x)}

    def _tactics_of(self, t):
        return self.data.tech.get(t, {}).get('tactics', [])

    def _platforms_of(self, t):
        return self.data.tech.get(t, {}).get('platforms', [])

    def groups(self):
        d = self.data
        out = []
        for g in d.names:
            prof = d.profiles[g]
            tac = {}
            for t in prof['techniques']:
                for x in self._tactics_of(t):
                    tac[x] = tac.get(x, 0) + 1
            plats = sorted({p for t in prof['techniques'] for p in self._platforms_of(t) if p != PRE_PLATFORM})
            out.append({'name': g, 'gid': self.gid.get(g, ''), 'attack': g in self.gid,
                        'aliases': self.aliases.get(g, []),
                        'sizes': {f: len(prof[f]) for f in d.features},
                        'tactics': tac, 'platforms': plats})
        return out

    def _pick_path(self, paths, rng):
        """One path among `paths` (sorted), preferring those whose CVE has an exploit recorded in the dataset."""
        with_exploit = [p for p in paths if self.data.exploits.get(p[-1], 0) > 0]
        pool = with_exploit or paths
        return pool[rng.randrange(len(pool))]

    # ------------------------------------------------------------------ the attack
    def generate(self, group, seed=0, platform='any', objective='both', tactics=None, include=None):
        d = self.data
        if group not in d.profiles:
            raise KeyError(f"unknown group {group}")
        prof = d.profiles[group]
        gid = self.gid.get(group)
        soft_of = d.tech_soft.get(group, {})
        chains = self.chains.get(group, {})
        techs = sorted(prof['techniques'])
        if platform and platform != 'any':
            techs = [t for t in techs if platform in self._platforms_of(t) or PRE_PLATFORM in self._platforms_of(t)]
        by_tactic = {}
        for t in techs:
            for tac in self._tactics_of(t):
                by_tactic.setdefault(tac, []).append(t)
        order = [t for t in TACTIC_ORDER if t in by_tactic and t not in OBJECTIVE_DROP.get(objective, set())]
        include = {lv: list(dict.fromkeys(str(x).strip() for x in (include or {}).get(lv, []) if str(x).strip()))
                   for lv in INCLUDE_LEVELS}
        if tactics is not None and set(tactics) == set(order):
            tactics = None                                   # the default tactics: same as not choosing any
        if tactics is None and not any(include.values()):
            return self._plain(group, gid, prof, techs, by_tactic, order, soft_of, chains, seed, platform, objective)
        return self._constrained(group, gid, prof, techs, by_tactic, order, soft_of, chains, seed, platform,
                                 objective, tactics, include)

    def _result(self, group, gid, prof, techs, seed, platform, objective, steps, tactics_drawn, constrained,
                included=None, not_placed=None):
        plan = {'mitigations': self._cover(steps, 'mitigations'), 'detections': self._cover(steps, 'detections')}
        return {'group': group, 'gid': gid or '', 'attack_group': gid is not None, 'seed': seed, 'platform': platform,
                'objective': objective, 'steps': steps, 'plan': plan,
                'n_techniques': len(prof['techniques']), 'n_candidates': len(techs),
                'tactics_drawn': tactics_drawn, 'constrained': constrained,
                'included': included or {lv: [] for lv in INCLUDE_LEVELS}, 'not_placed': not_placed or []}

    def _plain(self, group, gid, prof, techs, by_tactic, order, soft_of, chains, seed, platform, objective):
        """The plain generator: one technique of the profile for each tactic (unchanged behaviour)."""
        rng = random.Random(f"{seed}|{group}|{platform}|{objective}")
        used, steps = set(), []
        for tac in order:
            cand = [t for t in by_tactic[tac] if t not in used] or by_tactic[tac]
            t = cand[rng.randrange(len(cand))]            # each technique of the tactic equally likely
            used.add(t)
            step = self._step(group, gid, tac, t, rng, len(cand), 1.0 / len(cand))
            step['source'] = 'drawn'
            steps.append(step)
        return self._result(group, gid, prof, techs, seed, platform, objective, steps, order, False)

    def _constrained(self, group, gid, prof, techs, by_tactic, order, soft_of, chains, seed, platform, objective,
                     tactics, include):
        d = self.data
        drawn_tactics = order if tactics is None else [t for t in TACTIC_ORDER if t in set(tactics)]
        key = json.dumps({'tactics': None if tactics is None else sorted(tactics),
                          'include': {lv: include[lv] for lv in INCLUDE_LEVELS if include[lv]}}, sort_keys=True)
        rng = random.Random(f"{seed}|{group}|{platform}|{objective}|{key}")
        tset = set(drawn_tactics)

        def tactic_for(t):
            tl = [x for x in TACTIC_ORDER if x in self._tactics_of(t)]
            for x in tl:
                if x in tset:
                    return x
            return tl[0] if tl else None

        planned, not_placed, used = [], [], set()
        placed = {lv: [] for lv in INCLUDE_LEVELS}

        def new_step(t, chain=None):
            st = {'tactic': tactic_for(t), 'technique': t, 'software': [], 'chain': chain, 'because': []}
            planned.append(st)
            used.add(t)
            return st

        # 1. included techniques: a step each, under their tactic
        for t in include['techniques']:
            if t not in d.tech:
                not_placed.append({'level': 'techniques', 'id': t, 'reason': 'not a technique of ATT&CK release 16'})
                continue
            if tactic_for(t) is None:
                not_placed.append({'level': 'techniques', 'id': t, 'reason': 'ATT&CK gives it no tactic'})
                continue
            st = next((s for s in planned if s['technique'] == t), None) or new_step(t)
            st['because'].append({'level': 'techniques', 'id': t})
            placed['techniques'].append(t)

        # 2. included software: attached to a step whose technique it implements, or a step of its own
        for sw in include['software']:
            if sw not in d.soft:
                not_placed.append({'level': 'software', 'id': sw, 'reason': 'not a software of ATT&CK release 16'})
                continue
            impl = sorted(t for t in self.soft_techs.get(sw, ()) if tactic_for(t))
            if not impl:
                not_placed.append({'level': 'software', 'id': sw,
                                   'reason': 'no technique of this software in ATT&CK or in the records'})
                continue
            st = next((s for s in planned if s['technique'] in impl), None)
            if st is None:
                free = [t for t in impl if t not in used] or impl
                pref = [t for t in free if t in prof['techniques']] or free
                pref = [t for t in pref if tactic_for(t) in tset] or pref
                st = new_step(pref[rng.randrange(len(pref))])
            st['software'].append(sw)
            st['because'].append({'level': 'software', 'id': sw})
            placed['software'].append(sw)

        # 3. included CVEs, pulses, CWEs and CAPEC (most specific first): the attack path of a step that leads
        #    to them
        for lv in ('cve', 'pulse', 'cwe', 'capec'):
            for x in include[lv]:
                if lv == 'pulse':
                    def through(p, x=x):
                        return x in d.cve_pulses.get(p[3], ())
                else:
                    def through(p, x=x, k=PATH_INDEX[lv]):
                        return p[k] == x
                own = sorted(p for p in d.paths[group] if through(p))
                pool = [p for p in (own or [p for p in self.all_paths if through(p)]) if tactic_for(p[0])]
                if not pool:
                    not_placed.append({'level': lv, 'id': x,
                                       'reason': 'no technique leads to it in the records of the dataset'})
                    continue
                # a step whose path already goes through it
                st = next((s for s in planned if s['chain'] is not None
                           and through((s['technique'],) + tuple(s['chain']))), None)
                if st is not None:
                    st['because'].append({'level': lv, 'id': x, 'own_records': bool(own)})
                    placed[lv].append(x)
                    continue
                st = next((s for s in planned if s['chain'] is None and any(p[0] == s['technique'] for p in pool)),
                          None)
                if st is not None:
                    st['chain'] = self._pick_path([p for p in pool if p[0] == st['technique']], rng)[1:]
                else:
                    free = [p for p in pool if p[0] not in used] or pool
                    p = self._pick_path(free, rng)
                    st = next((s for s in planned if s['technique'] == p[0] and s['chain'] is None), None)
                    if st is None:
                        st = new_step(p[0], p[1:])
                    else:
                        st['chain'] = p[1:]
                st['because'].append({'level': lv, 'id': x, 'own_records': bool(own)})
                placed[lv].append(x)

        # 4. the ticked tactics that no included item covers: drawn from the group's profile
        covered = {s['tactic'] for s in planned}
        drawn = []
        for tac in drawn_tactics:
            if tac in covered or tac not in by_tactic:
                continue
            cand = [t for t in by_tactic[tac] if t not in used] or by_tactic[tac]
            t = cand[rng.randrange(len(cand))]            # each technique of the tactic equally likely
            used.add(t)
            drawn.append({'tactic': tac, 'technique': t, 'cand': len(cand), 'prob': 1.0 / len(cand)})

        # 5. the steps in kill-chain order (included steps first within a tactic)
        rank = {t: i for i, t in enumerate(TACTIC_ORDER)}
        items = [(rank.get(s['tactic'], 99), 0, k, 'in', s) for k, s in enumerate(planned)]
        items += [(rank.get(s['tactic'], 99), 1, k, 'draw', s) for k, s in enumerate(drawn)]
        steps = []
        for _, _, _, kind, s in sorted(items, key=lambda e: e[:3]):
            if kind == 'draw':
                step = self._step(group, gid, s['tactic'], s['technique'], rng, s['cand'], s['prob'])
                step['source'] = 'drawn'
            else:
                step = self._step(group, gid, s['tactic'], s['technique'], rng, None, None,
                                  chain=s['chain'], forced_software=s['software'])
                step['source'] = 'included'
                step['because'] = [dict(b, name=self.data.label(b['level'], b['id'])) for b in s['because']]
                step['in_profile'] = s['technique'] in prof['techniques']
                inc_pulses = [b['id'] for b in s['because'] if b['level'] == 'pulse']
                if inc_pulses and step['chain']:                 # show the included pulses first
                    allp = sorted(d.cve_pulses.get(step['chain']['cve']['id'], set()))
                    step['chain']['pulse_ids'] = (inc_pulses + [p for p in allp if p not in inc_pulses])[:3]
            steps.append(step)
        return self._result(group, gid, prof, techs, seed, platform, objective, steps, drawn_tactics, True,
                            placed, not_placed)

    def _step(self, group, gid, tactic, t, rng, n_cand, prob, chain=None, forced_software=None):
        d = self.data
        info = d.tech.get(t, {})
        procs = info.get('procedures', {})
        soft_ids = sorted(d.tech_soft.get(group, {}).get(t, set()))
        rng.shuffle(soft_ids)
        forced_software = list(forced_software or [])
        if forced_software:
            soft_ids = forced_software + [s for s in soft_ids if s not in forced_software]
        own_soft = set(d.profiles[group]['software'])          # included software may not be the group's
        how, how_source, how_by, how_own = None, None, None, True
        if gid and gid in procs and procs[gid]:
            how, how_source, how_by = procs[gid], 'group', group
        else:
            for s in soft_ids:
                if procs.get(s):
                    how, how_source, how_by = procs[s], 'software', d.soft.get(s, {}).get('name', s)
                    how_own = s in own_soft
                    break
        software = [{'id': s, 'name': d.soft.get(s, {}).get('name', s), 'type': d.soft.get(s, {}).get('type', ''),
                     'own': s in own_soft}
                    for s in soft_ids[:max(4, len(forced_software))]]
        lst = self.chains.get(group, {}).get(t, [])
        if chain is None and lst:
            with_exploit = [c for c in lst if d.exploits.get(c[2], 0) > 0]
            pool = with_exploit or lst
            chain = pool[rng.randrange(len(pool))]
        out_chain = None
        if chain:
            ca, w, v = chain
            pulses = sorted(d.cve_pulses.get(v, set()))
            cap = d.capec.get(ca, {})
            out_chain = {'capec': {'id': ca, 'name': cap.get('name', ''), 'severity': cap.get('severity', ''),
                                   'likelihood': cap.get('likelihood', '')},
                         'cwe': {'id': w, 'name': d.label('cwe', w)},
                         'cve': {'id': v, 'exploits': int(d.exploits.get(v, 0))},
                         'pulses': len(pulses), 'pulse_ids': pulses[:3],
                         'own_records': tuple(chain) in set(lst)}
        mit = [{'id': m, 'name': n} for m, n in sorted(info.get('mitigations', {}).items()) if m != PRE_COMPROMISE]
        det = [{'id': k.split(':', 1)[0], 'name': v} for k, v in sorted(info.get('detections', {}).items())]
        return {'tactic': tactic, 'technique': {'id': t, 'name': info.get('name', '')},
                'platforms': info.get('platforms', []), 'how': how, 'how_source': how_source, 'how_by': how_by,
                'how_own': how_own,
                'software': software, 'n_software': len(soft_ids), 'chain': out_chain, 'n_chains': len(lst),
                'n_capec': len({c[0] for c in lst}), 'n_cwe': len({c[1] for c in lst}), 'n_cve': len({c[2] for c in lst}),
                'mitigations': mit, 'detections': det, 'candidates': n_cand, 'probability': prob}

    @staticmethod
    def _cover(steps, key, n=5):
        items = {i: {x['id'] for x in s[key]} for i, s in enumerate(steps)}
        # detections are counted by ATT&CK data source (DSxxxx), so the plan names the data source, not a component
        names = {x['id']: (x['name'].split(':', 1)[0].strip() if key == 'detections' else x['name'])
                 for s in steps for x in s[key]}
        left, out = set(items), []
        while left and len(out) < n:
            best, cov = None, set()
            for c in sorted(names):
                hit = {i for i in left if c in items[i]}
                if len(hit) > len(cov):
                    best, cov = c, hit
            if not best:
                break
            out.append({'id': best, 'name': names[best], 'steps': sorted(cov)})
            left -= cov
        return {'chosen': out, 'uncovered': sorted(left)}
