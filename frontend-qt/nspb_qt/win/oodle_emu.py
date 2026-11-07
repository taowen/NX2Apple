#!/usr/bin/env python3
"""Windows replacement for the Mac host's Oodle / LZ5 decoder tool mode (`dm/host/hle_oodle.c`).

Supports the exact wire protocols expected by `shader_split.py`, `uepak.py`, and
`scripts/build_ipa_from_nsp.py`:
- `DM_OODLE_SERVE=1`:
  Reads `(comp_len: u64, raw_len: u64, comp_bytes)` requests from `stdin` and writes
  `(decoded_len: u64, raw_bytes)` responses to `stdout` (`0` on failure).
- `DM_OODLE_JOB=<job_file>`:
  Reads `"DMOJ"` (`0x4A4F4D44`), `count: u32`, and `count` records of
  `(comp_len: u32, raw_len: u32, comp_bytes)`, then writes `<job_file>.out` with `"DMOR"`
  (`0x524F4D44`), `count: u32`, and `count` records of `(decoded_len: u32, raw_bytes)`.

Performance optimizations over naive ARM64 emulation:
1. Native Windows x64 DLL Passthrough (`oo2core_*_win64.dll`):
   When a native Windows 64-bit Oodle DLL is available (`NSPB_OODLE_DLL`, `toolchain.json`,
   `%LOCALAPPDATA%\\NS Port Builder\\tools\\oodle`, or `PATH`), `OodleLZ_Decompress` calls execute
   natively on the host CPU via `ctypes` without starting Unicorn, falling back to the game's
   ARM64 decoder only if a block fails.
2. Single-Module Loading & Disk Pre-Relocated Image Cache:
   Only the module holding the decoder (`main`, or a `subsdk` if absent from `main`) is loaded,
   skipping the multi-megabyte Nintendo `sdk` NSO. Once relocated, the module snapshot and decoder
   offsets are cached in the temp directory keyed by `(path, size, mtime_ns)`, reducing worker
   startup from seconds to milliseconds.
3. In-VM ARM64 Fast Stubs (`arm64_emu.FAST_BASE`):
   `memcpy`, `memmove`, `memset`, `memcmp`, `strlen`, `malloc`, `calloc`, and `free` run as native
   ARM64 machine code inside the VM with zero Python `UC_HOOK_CODE` traps.
4. Pre-Allocated Fixed I/O Buffers & Pre-Packed Stack Frames:
   Input/output buffers and stack arguments are allocated once per `Decoder` instance and reused
   across all requests.
5. Multi-Core Parallel `DM_OODLE_JOB` Sharding:
   Large Unreal shader jobs (such as OCTOPATH TRAVELER 0's 10,596 shaders) are sharded across
   parallel worker processes that warm-start from the disk relocation cache.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

if __package__:
    from . import arm64_emu as E
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import arm64_emu as E  # type: ignore[no-redef]

_CACHE_VERSION = b'nspb-oodle-cache-v2'
_REDIRECT_SIGNATURES = (b'LZ5_decompress_safe', b'OodleLZ_Decompress', b'oodlelzcompressors.cpp')


def log(msg: str) -> None:
    print(f'[dm] {msg}', file=sys.stderr, flush=True)


def die(msg: str) -> None:
    log(msg)
    raise SystemExit(1)


def exefs_for(patched: str | Path) -> Path:
    """Locate the original (unpatched) ExeFS directory corresponding to `patched`."""
    for env_key in ('NSPB_EXEFS', 'NXW_EXEFS'):
        if os.environ.get(env_key):
            p = Path(os.environ[env_key])
            if p.is_file() and p.name == 'main':
                return p.parent
            if (p / 'main').is_file():
                return p
    p = Path(patched).resolve()
    if p.is_file() and p.name == 'main':
        return p.parent
    ver = p.name.removeprefix('patched_')
    for cand in (
        p.parent / 'extract' / ver / 'ExeFS',
        p.parent.parent / 'extract' / ver / 'ExeFS',
        p,
    ):
        if (cand / 'main').is_file():
            return cand
    die(f'oodle_emu: no extracted ExeFS for {p} (set NSPB_EXEFS)')
    raise AssertionError('unreachable')


def has_decoder(paths: list[str | Path] | str | Path) -> bool:
    """Fast static check: True if any module in `paths` embeds an Oodle or LZ5 decoder."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    checked: set[Path] = set()
    for item in paths:
        p = Path(item)
        cands: list[Path] = []
        if p.is_file():
            cands.append(p)
            for sub in sorted(p.parent.glob('subsdk*')):
                if sub.is_file():
                    cands.append(sub)
        elif p.is_dir():
            if (p / 'main').is_file():
                cands.append(p / 'main')
            for sub in sorted(p.glob('subsdk*')):
                if sub.is_file():
                    cands.append(sub)
        for mod_path in cands:
            try:
                rp = mod_path.resolve()
            except OSError:
                rp = mod_path
            if rp in checked:
                continue
            checked.add(rp)
            try:
                img, *_ = E.nso_load(mod_path)
            except Exception:
                try:
                    img = bytearray(mod_path.read_bytes())
                except OSError:
                    continue
            if any(sig in img for sig in _REDIRECT_SIGNATURES):
                return True
    return False


# ---- Native Windows x86_64 Oodle DLL Passthrough ----
def find_native_oodle_dll() -> Path | None:
    """Locate a native 64-bit Oodle DLL (`oo2core_*_win64.dll`) if present on the machine."""
    cands: list[Path] = []
    if env_dll := os.environ.get('NSPB_OODLE_DLL'):
        ep = Path(env_dll)
        if ep.is_file():
            cands.append(ep)
        elif ep.is_dir():
            cands.extend(sorted(ep.glob('oo2core_*_win64.dll'), reverse=True))
            cands.extend(sorted(ep.glob('oo2core_*.dll'), reverse=True))
    try:
        import toolchain as tc
        cfg = tc.load_config()
        if cfg_dll := cfg.get('oodle_dll'):
            cp = Path(cfg_dll)
            if cp.is_file():
                cands.append(cp)
            elif cp.is_dir():
                cands.extend(sorted(cp.glob('oo2core_*_win64.dll'), reverse=True))
        tools_oodle = tc.tools_root() / 'oodle'
        if tools_oodle.is_dir():
            cands.extend(sorted(tools_oodle.glob('oo2core_*_win64.dll'), reverse=True))
            cands.extend(sorted(tools_oodle.glob('oo2core_*.dll'), reverse=True))
    except Exception:
        pass
    for ver in (9, 8, 7, 6, 5):
        name = f'oo2core_{ver}_win64.dll'
        if found := shutil.which(name):
            cands.append(Path(found))
    for c in cands:
        if c.is_file():
            return c
    return None


class NativeOodleDll:
    """Wrapper around a native host `oo2core_*_win64.dll` exporting `OodleLZ_Decompress`."""

    def __init__(self, dll_path: Path) -> None:
        self.path = dll_path
        lib = ctypes.CDLL(str(dll_path))
        fn = getattr(lib, 'OodleLZ_Decompress', None)
        if fn is None:
            raise RuntimeError(f'{dll_path} does not export OodleLZ_Decompress')
        fn.restype = ctypes.c_int64
        fn.argtypes = [
            ctypes.c_char_p, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int,
        ]
        self._fn = fn
        self._scratch_n = 16 << 20
        self._scratch = ctypes.create_string_buffer(self._scratch_n)
        self._out_cap = 16 << 20
        self._out_buf = ctypes.create_string_buffer(self._out_cap)

    def decompress(self, comp: bytes, rl: int) -> bytes | None:
        if rl < 0:
            return None
        if rl == 0:
            return b''
        need_out = rl + 0x10000
        out_buf = self._out_buf if need_out <= self._out_cap else ctypes.create_string_buffer(need_out)
        got = self._fn(
            comp, len(comp),
            out_buf, rl,
            1, 0, 0,
            None, 0,
            None, None,
            self._scratch, self._scratch_n,
            3,
        )
        if got == rl:
            return ctypes.string_at(out_buf, rl)
        return None


# ---- Finding the decoder in an NSO module (`dm/host/hle_oodle.c`) ----
def eh_table(m: E.Module) -> list[int]:
    b = m.img
    m0 = m.mod0
    eh = struct.unpack_from('<i', b, m0 + 16)[0]
    ehs = m0 + eh
    penc, cenc, tenc = b[ehs + 1], b[ehs + 2], b[ehs + 3]
    o = [ehs + 4]

    def rd(enc: int) -> int:
        fmt, app, at = enc & 0xF, enc & 0x70, o[0]
        if fmt == 0x3:
            v = struct.unpack_from('<I', b, at)[0]
            o[0] += 4
        elif fmt == 0xB:
            v = struct.unpack_from('<i', b, at)[0]
            o[0] += 4
        elif fmt == 0x4:
            v = struct.unpack_from('<Q', b, at)[0]
            o[0] += 8
        else:
            die(f'oodle_emu: .eh_frame_hdr encoding {enc:#x} not handled')
            raise AssertionError('unreachable')
        return (v + at) if app == 0x10 else (v + ehs) if app == 0x30 else v

    rd(penc)
    n = rd(cenc)
    starts: list[int] = []
    for _ in range(n):
        starts.append(rd(tenc) & 0xFFFFFFFFFFFFFFFF)
        rd(tenc)
    return sorted(starts)


def func_start(starts: list[int], off: int) -> int:
    import bisect
    idx = bisect.bisect_right(starts, off)
    return starts[idx - 1] if idx > 0 else 0


def func_next(starts: list[int], off: int, text_end: int) -> int:
    import bisect
    idx = bisect.bisect_right(starts, off)
    return starts[idx] if idx < len(starts) else text_end


def str_fns(m: E.Module, starts: list[int], needle: str, whole: bool) -> list[int]:
    b = m.img
    ro, re_ = m.ro_off, m.data_off
    nb = needle.encode()
    strs: list[int] = []
    p = ro
    while len(strs) < 16:
        p = b.find(nb, p, re_)
        if p < 0:
            break
        s = p
        while s > ro and b[s - 1]:
            s -= 1
        if not whole or (s == p and p + len(nb) < len(b) and b[p + len(nb)] == 0):
            strs.append(s)
        p += len(nb)
    if not strs:
        return []
    want = set(strs)
    want_pages = {addr & ~0xFFF for addr in want}
    fns: list[int] = []
    te = m.text_end
    words = memoryview(b)[:te - te % 4].cast('I')
    n_words = len(words)
    for i in range(n_words):
        w = words[i]
        if (w & 0x9F000000) != 0x90000000:
            continue
        imm = ((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2)
        if imm & (1 << 20):
            imm -= 1 << 21
        page = ((i * 4) & ~0xFFF) + (imm << 12)
        if page not in want_pages:
            continue
        rd_reg = w & 31
        limit = min(n_words, i + 8)
        for k in range(i + 1, limit):
            w2 = words[k]
            if (w2 & 0xFF800000) != 0x91000000 or ((w2 >> 5) & 31) != rd_reg:
                continue
            if page + ((w2 >> 10) & 0xFFF) in want:
                f = func_start(starts, k * 4)
                if f not in fns:
                    fns.append(f)
            break
    return sorted(fns)


def _locate_decoder_in_module(mod: E.Module) -> tuple[str, int, tuple[int, int, int, int] | None] | None:
    if 'LZ5_decompress_safe' in mod.exports:
        return 'LZ5_decompress_safe', mod.exports['LZ5_decompress_safe'], None
    if b'OodleLZ_Decompress' not in mod.img and b'oodlelzcompressors.cpp' not in mod.img:
        return None
    starts = eh_table(mod)
    fns = str_fns(mod, starts, 'OodleLZ_Decompress', True)
    if len(fns) > 1:
        die(f'oodle_emu: {len(fns)} functions reference "OodleLZ_Decompress" in {mod.name} (expected 1)')
    if fns:
        return 'OodleLZ_Decompress', fns[0], None
    src = str_fns(mod, starts, 'oodlelzcompressors.cpp', False)
    leg = str_fns(
        mod,
        starts,
        'OODLE ERROR : OodleLZ_Decompress: asked to decode Legacy compressor',
        False,
    )
    if len(src) != 2 or len(leg) != 1 or leg[0] != src[0]:
        return None
    create = func_next(starts, src[0], mod.text_end)
    destroy = func_next(starts, create, mod.text_end)
    if func_next(starts, destroy, mod.text_end) != src[1]:
        return None
    base = mod.base
    lzd = (base + src[0], base + create, base + destroy, base + src[1])
    return 'OodleLZDecoder_DecodeSome', src[1], lzd


def _cache_file_for(mod_paths: list[tuple[str, str]]) -> Path | None:
    if os.environ.get('NSPB_OODLE_NO_CACHE'):
        return None
    h = hashlib.sha256(_CACHE_VERSION)
    h.update(E.FAST_STUB_BYTES)
    for name, p in mod_paths:
        try:
            st = os.stat(p)
        except OSError:
            return None
        h.update(f'{name}\0{os.path.realpath(p)}\0{st.st_size}\0{st.st_mtime_ns}\0'.encode())
    return Path(tempfile.gettempdir()) / f'nspb_oodle_{h.hexdigest()[:24]}.cache'


class Decoder:
    def __init__(self, patched: str | Path, *, use_fast_stubs: bool = True) -> None:
        self.patched = patched
        self.exefs = exefs_for(patched)
        self.use_fast_stubs = use_fast_stubs
        self.m: E.Machine | None = None
        self.kind: str = ''
        self.off: int = 0
        self.mod_name: str = 'main'
        self.fn: int = 0
        self.lzd: tuple[int, int, int, int] | None = None

        # Check for native Windows x64 Oodle DLL first
        self.native_dll: NativeOodleDll | None = None
        dll_path = find_native_oodle_dll()
        if dll_path is not None:
            try:
                self.native_dll = NativeOodleDll(dll_path)
            except Exception as e:
                log(f'oodle_emu: could not load native Oodle DLL {dll_path}: {e}')
                self.native_dll = None

        # If a native Oodle DLL is loaded and `main` contains Oodle (not LZ5), we can defer
        # initializing Unicorn until a block actually needs the emulated fallback!
        main_file = self.exefs / 'main'
        if self.native_dll is not None and main_file.is_file():
            try:
                img, *_ = E.nso_load(main_file)
                if b'LZ5_decompress_safe' not in img and (
                    b'OodleLZ_Decompress' in img or b'oodlelzcompressors.cpp' in img
                ):
                    self.kind = 'OodleLZ_Decompress'
                    self.off = 0
                    return
            except Exception:
                pass

        self._init_emu()

    @property
    def using_native_dll(self) -> bool:
        return self.native_dll is not None and self.kind != 'LZ5_decompress_safe'

    def _init_emu(self) -> None:
        if self.m is not None:
            return
        exefs = self.exefs
        main_pair = [('main', str(exefs / 'main'))] if (exefs / 'main').is_file() else []
        if not main_pair:
            die(f'oodle_emu: no main NSO in {exefs}')

        cache_path = _cache_file_for(main_pair) if self.use_fast_stubs else None
        if cache_path is not None and cache_path.is_file():
            try:
                raw = zlib.decompress(cache_path.read_bytes())
                snap = pickle.loads(raw)
                self.kind = snap['kind']
                self.off = snap['off']
                self.mod_name = snap['mod_name']
                self.lzd = snap['lzd']
                self.m = E.Machine([], use_fast_stubs=self.use_fast_stubs, pre_relocated=snap['vm'])
                target_mod = self.m.module(self.mod_name)
                self.fn = target_mod.base + self.off
                self._alloc_buffers()
                return
            except Exception:
                pass

        # Single-module load first: `main` holds the decoder in >99% of games.
        m = E.Machine(main_pair, use_fast_stubs=self.use_fast_stubs)
        main_mod = m.module('main')
        found = _locate_decoder_in_module(main_mod)
        target_mod = main_mod

        if found is None:
            # Check `subsdk*` only if `main` did not contain the decoder; never load `sdk`.
            sub_pairs = main_pair + [
                (p.name, str(p))
                for p in sorted(exefs.iterdir())
                if p.name.startswith('subsdk') and p.is_file()
            ]
            if len(sub_pairs) > len(main_pair):
                m = E.Machine(sub_pairs, use_fast_stubs=self.use_fast_stubs)
                for mod in m.mods:
                    if mod.name == 'main':
                        continue
                    found = _locate_decoder_in_module(mod)
                    if found is not None:
                        target_mod = mod
                        break

        if found is None:
            die(
                'oodle_emu: main has no "LZ5_decompress_safe", "OodleLZ_Decompress", or '
                'streaming Oodle decoder to find'
            )
            raise AssertionError('unreachable')

        self.m = m
        self.kind, self.off, self.lzd = found
        self.mod_name = target_mod.name
        self.fn = target_mod.base + self.off

        if self.kind == 'LZ5_decompress_safe':
            for name in ('malloc', 'free', 'calloc', 'realloc'):
                for mod in self.m.mods:
                    if name in mod.exports:
                        self.redirect(mod.base + mod.exports[name], name)

        if cache_path is not None and len(self.m.mods) == 1:
            try:
                snap = {
                    'kind': self.kind,
                    'off': self.off,
                    'mod_name': self.mod_name,
                    'lzd': self.lzd,
                    'vm': self.m.export_snapshot(),
                }
                blob = zlib.compress(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL), level=1)
                tmp_path = cache_path.with_suffix(f'.tmp.{os.getpid()}')
                tmp_path.write_bytes(blob)
                os.replace(tmp_path, cache_path)
            except Exception:
                pass

        self._alloc_buffers()

    def _alloc_buffers(self) -> None:
        assert self.m is not None
        m = self.m
        self.scratch_n = 16 << 20
        self.scratch = m.malloc(self.scratch_n)
        self.in_cap = 8 << 20
        self.in_buf = m.malloc(self.in_cap)
        self.out_cap = 16 << 20
        self.out_buf = m.malloc(self.out_cap)
        self.lzd_out = m.malloc(16)
        self.mark = m.heap_mark()
        self._oodle_stack = struct.pack('<6Q', 0, 0, 0, self.scratch, self.scratch_n, 3)

    def redirect(self, addr: int, name: str) -> None:
        """Overwrite the function at `addr` with an indirect jump to the emulator's handler."""
        assert self.m is not None
        target = self.m.resolve_import(name)
        patch = struct.pack('<IIQ', 0x58000050, 0xD61F0200, target)
        self.m.write(addr, patch)
        for mod in self.m.mods:
            if mod.base <= addr < mod.base + len(mod.img):
                off = addr - mod.base
                mod.img[off:off + len(patch)] = patch

    def run(self, comp: bytes, rl: int) -> bytes | None:
        if rl < 0:
            return None
        if rl == 0:
            return b''
        if self.native_dll is not None and self.kind != 'LZ5_decompress_safe':
            res = self.native_dll.decompress(comp, rl)
            if res is not None:
                return res
            if self.m is None:
                self._init_emu()

        assert self.m is not None
        m = self.m
        m.heap_reset(self.mark)
        cl = len(comp)
        c = self.in_buf if cl <= self.in_cap else m.malloc(cl or 1)
        r = self.out_buf if (rl + 0x10000) <= self.out_cap else m.malloc(rl + 0x10000)
        if comp:
            m.write(c, comp)

        if self.kind == 'LZ5_decompress_safe':
            got = m.call_fast(self.fn, c, r, cl, rl) if cl < 2 ** 31 and rl < 2 ** 31 else -1
            got = got - (1 << 64) if got >= 1 << 63 else got
        elif self.kind == 'OodleLZ_Decompress':
            got = m.call_fast(self.fn, c, cl, r, rl, 1, 0, 0, 0, self._oodle_stack)
            got = got - (1 << 64) if got >= 1 << 63 else got
        else:
            got = self.lzd_run(c, comp, cl, r, rl)
        return m.read(r, rl) if got == rl else None

    def lzd_run(self, c: int, comp: bytes, cl: int, r: int, rl: int) -> int:
        assert self.m is not None and self.lzd is not None
        size, create, destroy, some = self.lzd
        if cl < 2 or (comp[0] & 0x0F) != 0x0C:
            return -1
        compressor = {6: 8, 10: 9, 12: 13}.get(comp[1] & 0x7F)
        if compressor is None:
            return -1
        m = self.m
        need = m.call_fast(size, compressor, rl) & 0xFFFFFFFF
        if need <= 0 or need >= 1 << 31:
            return -1
        mem = self.scratch if need + 64 <= self.scratch_n else m.malloc(need + 64)
        dec = m.call_fast(create, compressor, rl, mem, need + 64)
        if not dec:
            return -1
        out = self.lzd_out
        pos = cpos = 0
        stack_some = struct.pack('<3Q', 0, 0, 3)
        while pos < rl:
            m.write(out, b'\0' * 16)
            ok = (
                m.call_fast(
                    some,
                    dec,
                    out,
                    r,
                    pos,
                    rl,
                    rl - pos,
                    c + cpos,
                    cl - cpos,
                    stack_bytes=struct.pack('<4Q', 1, 0, 0, 3),
                )
                & 0xFFFFFFFF
            )
            done, used = struct.unpack('<ii', m.read(out, 8))
            if not ok or (done <= 0 and used <= 0):
                break
            pos += done
            cpos += used
        del stack_some
        m.call_fast(destroy, dec)
        return pos


def read_all(f, n: int) -> bytes | None:
    b = bytearray()
    while len(b) < n:
        k = f.read(n - len(b))
        if not k:
            return None
        b += k
    return bytes(b)


def serve(dec: Decoder) -> int:
    i, o = sys.stdin.buffer, sys.stdout.buffer
    n = bad = 0
    mode_desc = (
        f'native {dec.native_dll.path.name}'
        if dec.using_native_dll
        else f'emulated ARM64 at {dec.mod_name}+{dec.off:#x}'
    )
    log(f'DM_OODLE_SERVE: {dec.kind} ({mode_desc}, win/oodle_emu.py), requests on stdin')
    while True:
        h = read_all(i, 16)
        if h is None:
            break
        cl, rl = struct.unpack('<QQ', h)
        if cl > 1 << 32 or rl > 1 << 32:
            die(f'DM_OODLE_SERVE: request too large ({cl:#x} -> {rl:#x})')
        comp = read_all(i, cl)
        if comp is None:
            die('DM_OODLE_SERVE: request cut short')
        try:
            out = dec.run(comp, rl)
        except E.EmuError as e:
            out = None
            log(f'[RED] DM_OODLE_SERVE: request {n}: {e}')
        if out is None and rl:
            bad += 1
            if bad <= 8:
                log(f'[RED] DM_OODLE_SERVE: request {n} ({cl} -> {rl} bytes) failed')
        o.write(struct.pack('<Q', len(out) if out else 0))
        if out:
            o.write(out)
        o.flush()
        n += 1
    log(f'DM_OODLE_SERVE: {n} requests, {bad} failed')
    return 1 if bad else 0


def _read_job_records(path: str | Path) -> list[tuple[int, int, bytes]]:
    with open(path, 'rb') as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            die(f'DM_OODLE_JOB: {path} is truncated')
        magic, n = struct.unpack('<II', hdr)
        if magic != 0x4A4F4D44:
            die(f'DM_OODLE_JOB: {path} is not a job file')
        records: list[tuple[int, int, bytes]] = []
        for k in range(n):
            rh = f.read(8)
            if len(rh) != 8:
                die(f'DM_OODLE_JOB: record {k} header ends early')
            cl, rl = struct.unpack('<II', rh)
            comp = f.read(cl)
            if len(comp) != cl:
                die(f'DM_OODLE_JOB: record {k} ends early')
            records.append((cl, rl, comp))
    return records


def _write_job_shard(path: Path, records: list[tuple[int, int, bytes]]) -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', 0x4A4F4D44, len(records)))
        for cl, rl, comp in records:
            f.write(struct.pack('<II', cl, rl))
            f.write(comp)


def job(dec: Decoder, path: str) -> int:
    records = _read_job_records(path)
    n = len(records)
    mode_desc = (
        f'native {dec.native_dll.path.name}'
        if dec.using_native_dll
        else f'emulated ARM64 at {dec.mod_name}+{dec.off:#x}'
    )
    log(f'DM_OODLE_JOB: {dec.kind} ({mode_desc}, win/oodle_emu.py), {n} records')

    # Multi-core parallel sharding for large emulated jobs (spawning Python + Unicorn costs ~150 ms
    # on Windows, so auto-sharding kicks in only when the job is large enough to amortize spawn cost,
    # or when NSPB_OODLE_WORKERS is explicitly set).
    max_w_env = os.environ.get('NSPB_OODLE_WORKERS')
    explicit_w = int(max_w_env) if (max_w_env and max_w_env.isdigit()) else 0
    total_raw_bytes = sum(rl for _cl, rl, _comp in records)
    should_shard = (
        not dec.using_native_dll
        and (
            (explicit_w >= 2 and n >= 16)
            or (explicit_w == 0 and n >= 512 and total_raw_bytes >= (4 << 20))
        )
    )
    max_w = explicit_w if explicit_w >= 2 else min(os.cpu_count() or 4, 8)
    if should_shard and max_w >= 2:
        workers = min(max_w, max(2, n // 16))
        shard_paths: list[Path] = []
        procs: list[tuple[int, Path, subprocess.Popen]] = []
        try:
            # Partition records evenly by raw output bytes so each worker does similar work
            buckets: list[list[tuple[int, int, int, bytes]]] = [[] for _ in range(workers)]
            loads = [0] * workers
            for idx, (cl, rl, comp) in enumerate(records):
                target = min(range(workers), key=loads.__getitem__)
                buckets[target].append((idx, cl, rl, comp))
                loads[target] += max(rl, 256)

            buckets = [b for b in buckets if b]
            if len(buckets) >= 2:
                env = dict(os.environ, NSPB_OODLE_WORKERS='1')
                for s_idx, b in enumerate(buckets):
                    sp = Path(f'{path}.shard.{os.getpid()}.{s_idx}')
                    shard_paths.append(sp)
                    _write_job_shard(sp, [(cl, rl, comp) for _i, cl, rl, comp in b])
                    s_env = dict(env, DM_OODLE_JOB=str(sp))
                    p = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), str(dec.patched)],
                        env=s_env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    procs.append((s_idx, sp, p))

                all_ok = True
                results: dict[int, bytes | None] = {}
                for s_idx, sp, p in procs:
                    rc = p.wait()
                    out_sp = Path(str(sp) + '.out')
                    shard_paths.append(out_sp)
                    if rc != 0 or not out_sp.is_file():
                        all_ok = False
                        break
                    with open(out_sp, 'rb') as sf:
                        smagic, sn = struct.unpack('<II', sf.read(8))
                        if smagic != 0x524F4D44 or sn != len(buckets[s_idx]):
                            all_ok = False
                            break
                        for orig_idx, _cl, rl, _comp in buckets[s_idx]:
                            got_len = struct.unpack('<I', sf.read(4))[0]
                            results[orig_idx] = sf.read(got_len) if got_len else None

                if all_ok and len(results) == n:
                    bad = total = 0
                    with open(path + '.out', 'wb') as out:
                        out.write(struct.pack('<II', 0x524F4D44, n))
                        for k in range(n):
                            r = results[k]
                            if r is None:
                                bad += 1
                                out.write(struct.pack('<I', 0))
                            else:
                                out.write(struct.pack('<I', len(r)))
                                out.write(r)
                                total += len(r)
                    log(
                        f'DM_OODLE_JOB: {n - bad} records decoded across {len(buckets)} workers '
                        f'({bad} failed, {total / 1048576:.1f} MB out) -> {path}.out'
                    )
                    return 1 if bad else 0
        except Exception:
            pass
        finally:
            for _, _, p in procs:
                if p.poll() is None:
                    try:
                        p.kill()
                        p.wait()
                    except Exception:
                        pass
            for sp in shard_paths:
                try:
                    sp.unlink(missing_ok=True)
                except OSError:
                    pass

    with open(path + '.out', 'wb') as out:
        out.write(struct.pack('<II', 0x524F4D44, n))
        bad = total = 0
        for k, (cl, rl, comp) in enumerate(records):
            try:
                r = dec.run(comp, rl)
            except E.EmuError as e:
                r = None
                log(f'[RED] DM_OODLE_JOB: record {k}: {e}')
            if r is None:
                bad += 1
                if bad <= 8:
                    log(f'[RED] DM_OODLE_JOB: record {k} ({cl} -> {rl} bytes) failed')
                out.write(struct.pack('<I', 0))
            else:
                out.write(struct.pack('<I', rl))
                out.write(r)
                total += rl
            if k and k % 2000 == 0:
                log(f'DM_OODLE_JOB: {k} / {n} records')
    log(
        f'DM_OODLE_JOB: {n - bad} records decoded with the game\'s {dec.kind} '
        f'({bad} failed, {total / 1048576:.1f} MB out) -> {path}.out'
    )
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        die('usage: oodle_emu.py <rewritten modules folder>   (DM_OODLE_SERVE=1 or DM_OODLE_JOB=<file>)')
    if os.environ.get('DM_OODLE_JOB'):
        return job(Decoder(args[0]), os.environ['DM_OODLE_JOB'])
    if os.environ.get('DM_OODLE_SERVE'):
        return serve(Decoder(args[0]))
    die(
        'oodle_emu: neither DM_OODLE_SERVE nor DM_OODLE_JOB is set: the emulator runs only the '
        'decoder tool mode (booting a game needs a Mac with Apple silicon)'
    )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
