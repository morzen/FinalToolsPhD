"""Minimal reader for mongodump .bson files (no pymongo needed). Read-only; enough BSON types for the
MITRE ATT&CK, CAPEC and CWE collections of Cyber_-Necronomicon."""
import struct
import datetime


def _cstring(b, i):
    j = b.index(b'\x00', i)
    return b[i:j].decode('utf-8', 'replace'), j + 1


def _doc(b, i, as_list=False):
    size = struct.unpack_from('<i', b, i)[0]
    end = i + size - 1
    i += 4
    out = [] if as_list else {}
    while i < end:
        t = b[i]
        i += 1
        name, i = _cstring(b, i)
        if t == 0x01:
            v = struct.unpack_from('<d', b, i)[0]
            i += 8
        elif t == 0x02:
            n = struct.unpack_from('<i', b, i)[0]
            v = b[i + 4:i + 4 + n - 1].decode('utf-8', 'replace')
            i += 4 + n
        elif t == 0x03:
            v, i = _doc(b, i)
        elif t == 0x04:
            v, i = _doc(b, i, True)
        elif t == 0x05:
            n = struct.unpack_from('<i', b, i)[0]
            v = b[i + 5:i + 5 + n]
            i += 5 + n
        elif t == 0x07:
            v = b[i:i + 12].hex()
            i += 12
        elif t == 0x08:
            v = b[i] == 1
            i += 1
        elif t == 0x09:
            ms = struct.unpack_from('<q', b, i)[0]
            i += 8
            try:
                v = datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=ms)
            except OverflowError:
                v = ms
        elif t == 0x0A:
            v = None
        elif t == 0x10:
            v = struct.unpack_from('<i', b, i)[0]
            i += 4
        elif t == 0x11:
            v = struct.unpack_from('<Q', b, i)[0]
            i += 8
        elif t == 0x12:
            v = struct.unpack_from('<q', b, i)[0]
            i += 8
        elif t == 0x13:
            v = b[i:i + 16]
            i += 16
        else:
            raise ValueError(f'BSON type {t:#x} not handled')
        if as_list:
            out.append(v)
        else:
            out[name] = v
    return out, end + 1


def iter_docs(path):
    """Every document of a mongodump .bson file, in file order."""
    with open(path, 'rb') as fh:
        data = fh.read()
    i = 0
    while i < len(data):
        d, i = _doc(data, i)
        yield d
