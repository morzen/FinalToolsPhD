#!/usr/bin/env python3
"""
build_config_tool.py -- make a throw-away copy of the six-feature tool for ONE feature configuration.

The copy holds the tool's own code files, byte for byte (common/protocol.py, every method's query.py / train.py,
the CTI template), and a BaseData/EncodedData.csv in which every feature type outside the configuration is set to
-1, the thesis' code for a missing value.  All rows are kept (the Siamese network samples rows), only the values
of the blanked columns change.  The 100 MB mapping file is linked, not copied.  Nothing in the tool folder is
written.  The train.py of each learned method can then be run inside the copy, unchanged.

  <venv>/bin/python build_config_tool.py --tool-root ~/Desktop/claudePhDFolder/FinalToolsPhD_6feat \
          --config T --dest <folder>
"""
import os
import sys
import json
import shutil
import hashlib
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
from sweep_core import cfg_families, cfg_key, FAMILIES          # noqa: E402

CODE_FILES = [
    'common/__init__.py', 'common/protocol.py',
    '1.XGboost/train.py', '1.XGboost/query.py',
    '2.GNN/train.py', '2.GNN/query.py',
    '3.SiameseNetwork/train.py', '3.SiameseNetwork/query.py',
    '4.Attention-Based_MIL/train.py', '4.Attention-Based_MIL/query.py',
    '5.RF_FeatureEngineering/train.py', '5.RF_FeatureEngineering/query.py',
    'Jaccards_Correlation/Jaccards_correlation.py', 'featureMatching/featureMatching.py',
    'TVD_Total_Variation_Distance/TVD_featureIng.py',
    'CTI_eval/CTI_Eval_empty_template.json',
]
COLUMNS = {'techniques': 'techniques_str', 'software': 'software_str', 'capec': 'CAPEC_ID', 'cwe': 'CWE_ID',
           'cve': 'CVE_ID', 'pulse': 'Pulse_ID'}          # = common/protocol.py COLUMNS


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(tool_root, key, dest):
    import pandas as pd
    tool_root, dest = os.path.abspath(tool_root), os.path.abspath(dest)
    S = cfg_families(key)
    if os.path.commonpath([tool_root, dest]) == tool_root:
        sys.exit("refusing to write inside the tool folder")
    os.makedirs(dest, exist_ok=True)
    copied = {}
    for rel in CODE_FILES:
        src = os.path.join(tool_root, rel)
        if not os.path.exists(src):
            if rel.endswith('__init__.py'):
                os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
                open(os.path.join(dest, rel), 'w').close()
                continue
            sys.exit(f"missing in the tool folder: {rel}")
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied[rel] = md5(dst)
    base = os.path.join(dest, 'BaseData')
    os.makedirs(base, exist_ok=True)
    src_map = os.path.join(tool_root, 'BaseData', 'comparison_Encoded_df.csv')
    dst_map = os.path.join(base, 'comparison_Encoded_df.csv')
    if not os.path.exists(dst_map):
        try:
            os.symlink(src_map, dst_map)
        except OSError:
            shutil.copy2(src_map, dst_map)
    src_enc = os.path.join(tool_root, 'BaseData', 'EncodedData.csv')
    dst_enc = os.path.join(base, 'EncodedData.csv')
    df = pd.read_csv(src_enc)
    for f in FAMILIES:
        if f not in S:
            df[COLUMNS[f]] = -1
    tmp = dst_enc + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, dst_enc)
    kept = {f: int((df[COLUMNS[f]] != -1).sum()) for f in FAMILIES}
    info = {'config': cfg_key(S), 'families': list(S), 'tool_root': tool_root,
            'source_EncodedData_md5': md5(src_enc), 'rows': int(len(df)), 'non_missing_values_kept': kept,
            'code_files_md5': copied,
            'note': 'Copy of the six-feature tool. Code files are byte-identical copies; in BaseData/EncodedData.csv '
                    'every feature type outside the configuration is -1 (missing). The mapping file is a link.'}
    with open(os.path.join(dest, 'CONFIG.json'), 'w') as fh:
        json.dump(info, fh, indent=1)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool-root', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--dest', required=True)
    a = ap.parse_args()
    info = build(a.tool_root, a.config, a.dest)
    print(f"[build] {info['config']} -> {a.dest}: {info['rows']} rows, kept values {info['non_missing_values_kept']}")


if __name__ == '__main__':
    main()
