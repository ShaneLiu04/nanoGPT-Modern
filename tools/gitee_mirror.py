#!/usr/bin/env python3
"""Gitee slim-mirror sync (corporate proxy POST cap ~100KB per request).

The corporate BDWAF blocks gitee HTTPS git pushes whose pack body exceeds
~100KB.  A handful of blobs in this repository compress to more than that
(uv.lock ~400KB, three doc PNGs 102-180KB), so the full-fidelity history can
never be transported over this channel.

This tool therefore maintains a parallel *slim mirror* commit chain in
``<repo>-gitee-mirror/``:

- untransportable blobs (gzip size > 95KB) are replaced: PNGs are palette-
  quantized down to <=58KB, ``uv.lock`` becomes a short stub file;
- every other object (and the full commit metadata: author, dates, message)
  is preserved verbatim, so the mirror chain mirrors 1:1 except for those
  paths;
- the local history is NEVER modified.

Pushes are batched: each HTTP POST carries at most ~70KB of pack data
(blobs are pre-seeded via synthetic commits when a single commit would
exceed the budget), following the ops discipline established for this
proxy.

Usage::

    python tools/gitee_mirror.py            # build/sync mirror and push
    python tools/gitee_mirror.py --no-push  # build/sync mirror only
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import subprocess
import sys

MAIN_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR_DIR = MAIN_REPO + "-gitee-mirror"
REMOTE_NAME = "gitee"
REMOTE_URL = "https://gitee.com/liu-xingyan04/nano-gpt-modern.git"
GZ_LIMIT = 95 * 1024
PNG_TARGET = 58 * 1024
PACK_BUDGET_KB = 70.0
GZ_PROBE_THRESHOLD = 40 * 1024

STUB_LOCK = (
    b"# Slim-mirror stub (gitee): the real uv.lock (~1.4MB, ~400KB gzipped)\n"
    b"# exceeds the corporate proxy's ~100KB POST cap and cannot be hosted\n"
    b"# on this remote.  Regenerate locally with `uv lock`, or copy the file\n"
    b"# from the GitHub origin mirror.\n"
)


def die(msg: str) -> None:
    print(f"[gitee-mirror] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def mgit(*args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
    """git invocation inside the mirror repo (shares objects via alternates)."""
    r = subprocess.run(
        ["git", "-C", MIRROR_DIR, *args], input=input_bytes, capture_output=True
    )
    if check and r.returncode != 0:
        raise RuntimeError(
            "git %s failed: %s" % (args, r.stderr.decode(errors="replace")[:400])
        )
    return r.stdout


def push(refspec: str) -> tuple[bool, str]:
    r = subprocess.run(
        ["git", "-C", MIRROR_DIR, "push", REMOTE_NAME, refspec], capture_output=True
    )
    out = (r.stdout + r.stderr).decode(errors="replace")
    return r.returncode == 0, out


def gz_size(data: bytes) -> int:
    return len(gzip.compress(data, 9))


def init_mirror() -> None:
    if not os.path.isdir(os.path.join(MIRROR_DIR, ".git")):
        os.makedirs(MIRROR_DIR, exist_ok=True)
        subprocess.run(["git", "init", MIRROR_DIR], capture_output=True, check=True)
        alt_dir = os.path.join(MIRROR_DIR, ".git", "objects", "info")
        os.makedirs(alt_dir, exist_ok=True)
        with open(os.path.join(alt_dir, "alternates"), "w", encoding="utf-8", newline="\n") as f:
            f.write(os.path.join(MAIN_REPO, ".git", "objects").replace("\\", "/") + "\n")
        subprocess.run(
            ["git", "-C", MIRROR_DIR, "remote", "add", REMOTE_NAME, REMOTE_URL],
            capture_output=True,
        )
    remotes = subprocess.run(
        ["git", "-C", MIRROR_DIR, "remote"], capture_output=True, text=True
    ).stdout.split()
    if REMOTE_NAME not in remotes:
        subprocess.run(
            ["git", "-C", MIRROR_DIR, "remote", "add", REMOTE_NAME, REMOTE_URL],
            capture_output=True,
            check=True,
        )


def gg(*args: str, input_bytes: bytes | None = None, check: bool = True) -> bytes:
    """git invocation inside the main repo (read side)."""
    r = subprocess.run(
        ["git", "-C", MAIN_REPO, *args], input=input_bytes, capture_output=True
    )
    if check and r.returncode != 0:
        raise RuntimeError(
            "git %s failed: %s" % (args, r.stderr.decode(errors="replace")[:400])
        )
    return r.stdout


def scan_replacements() -> dict[str, str]:
    """Return {original blob sha: replacement blob sha} for fat blobs."""
    objs = gg("rev-list", "--objects", "HEAD").decode(errors="replace")
    path_of: dict[str, str] = {}
    shas = []
    for line in objs.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        shas.append(parts[0])
        if len(parts) == 2:
            path_of.setdefault(parts[0], parts[1])
    batch = mgit(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_bytes=("\n".join(shas) + "\n").encode(),
    ).decode(errors="replace")

    repl: dict[str, str] = {}
    for line in batch.strip().splitlines():
        sha, otype, size = line.split()
        if otype != "blob" or int(size) <= GZ_PROBE_THRESHOLD:
            continue
        blob = mgit("cat-file", "blob", sha)
        path = path_of.get(sha, "")
        is_img = path.lower().endswith((".png", ".jpg", ".jpeg"))
        # transport size: images ride the wire ~raw (already compressed),
        # everything else packs down to roughly its gzipped size
        wire = max(len(blob), gz_size(blob)) if is_img else gz_size(blob)
        if wire > GZ_LIMIT or (is_img and wire > PNG_TARGET):
            if os.path.basename(path) == "uv.lock":
                new_blob = STUB_LOCK
            elif is_img:
                new_blob = quantize_png(blob, path)
            else:
                die(
                    f"untransportable blob {sha[:10]} at '{path}' "
                    f"({int(size)//1024}KB) has no replacement rule; add one"
                )
            new_sha = mgit(
                "hash-object", "-w", "--stdin", input_bytes=new_blob
            ).decode().strip()
            repl[sha] = new_sha
            print(
                f"  replace {path}: {int(size)//1024}KB -> "
                f"{len(new_blob)//1024}KB (gz {gz_size(new_blob)//1024}KB)"
            )
    return repl


def quantize_png(blob: bytes, path: str) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        die("Pillow required to shrink PNGs for the gitee mirror (pip install pillow)")

    img = Image.open(io.BytesIO(blob))
    w, h = img.size
    for colors, maxw in [(256, None), (256, 1600), (128, 1400), (64, 1100), (32, 900)]:
        im = img
        if maxw and w > maxw:
            im = img.resize((maxw, int(h * maxw / w)), Image.LANCZOS)
        if im.mode not in ("P", "L"):
            im = im.convert("RGB").quantize(
                colors=colors, method=Image.MEDIANCUT, dither=Image.NONE
            )
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        if gz_size(buf.getvalue()) <= PNG_TARGET:
            return buf.getvalue()
    die(f"cannot shrink {path} below {PNG_TARGET//1024}KB even at 32 colors/900px")


class MirrorChain:
    """Rebuilds the commit chain with seed commits woven in.

    Each mirrored commit is rebased onto a chain of synthetic "seed"
    commits (one per ~42KB blob group), so that pushing the chain
    element-by-element keeps every HTTP POST under the proxy cap:
    git negotiates packs by commit ancestry, so the seeds must be
    ancestors of the mirrored commits they feed.
    """

    def __init__(self, repl: dict[str, str]):
        self.repl = repl
        self.tree_memo: dict[str, str] = {}
        self.commit_map: dict[str, str] = {}
        self.push_seq: list[str] = []
        self.blob_kb: dict[str, float] = {}

    def rebuild_tree(self, tree_sha: str) -> str:
        if tree_sha in self.tree_memo:
            return self.tree_memo[tree_sha]
        raw = mgit("ls-tree", "-z", tree_sha)
        entries = [e for e in raw.split(b"\x00") if e]
        new_entries = []
        changed = False
        for e in entries:
            meta, name = e.split(b"\t", 1)
            mode, otype, sha = meta.decode().split(" ")
            if otype == "blob" and sha in self.repl:
                sha = self.repl[sha]
                changed = True
            elif otype == "tree":
                sub = self.rebuild_tree(sha)
                if sub != sha:
                    changed = True
                sha = sub
            new_entries.append(f"{mode} {otype} {sha}".encode() + b"\t" + name)
        if not changed:
            self.tree_memo[tree_sha] = tree_sha
            return tree_sha
        new_sha = (
            mgit("mktree", "-z", input_bytes=b"\x00".join(new_entries) + b"\x00")
            .decode()
            .strip()
        )
        self.tree_memo[tree_sha] = new_sha
        return new_sha

    def _build_tree_on(self, prev: str | None, batch: list[tuple[str, str, str]]) -> str:
        tmp_index = os.path.join(MIRROR_DIR, ".git", "seed_index")
        env = dict(os.environ, GIT_INDEX_FILE=tmp_index)
        if prev is None:
            subprocess.run(
                ["git", "-C", MIRROR_DIR, "read-tree", "--empty"],
                env=env, capture_output=True, check=True,
            )
        else:
            subprocess.run(
                ["git", "-C", MIRROR_DIR, "read-tree", prev],
                env=env, capture_output=True, check=True,
            )
        index_info = "".join(f"{mode} {sha} 0\t{path}\n" for mode, sha, path in batch)
        subprocess.run(
            ["git", "-C", MIRROR_DIR, "update-index", "--index-info"],
            env=env, input=index_info.encode(), capture_output=True, check=True,
        )
        return subprocess.run(
            ["git", "-C", MIRROR_DIR, "write-tree"], env=env, capture_output=True, text=True
        ).stdout.strip()

    def _blob_kb(self, shas: list[str]) -> dict[str, float]:
        """max(raw, gz) size in KB for each blob, batched + cached."""
        todo = [s for s in shas if s not in self.blob_kb]
        if todo:
            data = mgit("cat-file", "--batch", input_bytes=("\n".join(todo) + "\n").encode())
            pos = 0
            for _ in todo:
                nl = data.index(b"\n", pos)
                parts = data[pos:nl].decode().split()
                if parts[1] == "missing":
                    pos = nl + 1
                    continue
                size = int(parts[2])
                body = data[nl + 1:nl + 1 + size]
                self.blob_kb[parts[0]] = len(gzip.compress(body, 9)) / 1024.0
                pos = nl + 1 + size + 1
        return {s: self.blob_kb[s] for s in shas}

    def _commit_tree(self, tree: str, prev: str | None, message: bytes, env_meta: dict | None = None) -> str:
        args = ["git", "-C", MIRROR_DIR, "commit-tree", tree]
        if prev is not None:
            args += ["-p", prev]
        r = subprocess.run(args, input=message, capture_output=True, env=env_meta)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace")[:300])
        return r.stdout.decode().strip()

    def mirror_commit(self, orig: str) -> str:
        if orig in self.commit_map:
            return self.commit_map[orig]
        raw = gg("cat-file", "commit", orig)
        lines = raw.split(b"\n")
        orig_tree = None
        orig_parents: list[str] = []
        header_meta: dict[str, bytes] = {}
        for line in lines:
            if line.startswith(b"tree "):
                orig_tree = line[5:].decode()
            elif line.startswith(b"parent "):
                orig_parents.append(line[7:].decode())
            elif line.startswith(b"author ") or line.startswith(b"committer "):
                kind = line.split(b" ", 1)[0].decode()
                header_meta[kind] = line[len(kind) + 1:]
            elif line == b"":
                break
        if orig_tree is None:
            die(f"cannot parse commit {orig[:8]}")

        # preserve original author/committer identity and timestamps
        ident_re = re.compile(rb"^(.*?) <(.*?)> (\d+) ([+-]\d{4})$")
        env_meta = dict(os.environ)
        for kind, keys in (
            ("author", ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE")),
            ("committer", ("GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE")),
        ):
            m = ident_re.match(header_meta.get(kind, b""))
            if m:
                name, email, ts, tz = m.groups()
                env_meta[keys[0]] = name.decode()
                env_meta[keys[1]] = email.decode()
                env_meta[keys[2]] = f"@{ts.decode()} {tz.decode()}"

        prev = self.commit_map[orig_parents[0]] if orig_parents else None

        # objects newly introduced by this commit (vs its parents)
        rev_args = ["rev-list", "--objects", orig] + [f"^{p}" for p in orig_parents]
        new_objs = gg(*rev_args).decode(errors="replace").strip().splitlines()
        new_shas = {l.split()[0] for l in new_objs if l.strip()}

        # new blob entries in the mirrored tree, split into budget groups;
        # replaced blobs map to their slim versions
        entries = []
        for line in mgit("ls-tree", "-r", "-z", orig_tree).split(b"\x00"):
            if not line:
                continue
            meta, path = line.split(b"\t", 1)
            mode, otype, sha = meta.decode().split(" ")
            if otype != "blob":
                continue
            msha = self.repl.get(sha, sha)
            if msha == sha and sha not in new_shas:
                continue
            entries.append((mode, msha, path.decode()))
        sizes = self._blob_kb([msha for _, msha, _ in entries])
        entries = [(mode, msha, path, sizes[msha]) for mode, msha, path in entries]

        batches, cur, cur_kb = [], [], 0.0
        for mode, msha, path, kb in entries:
            if cur and cur_kb + kb > PACK_BUDGET_KB * 0.6:
                batches.append(cur)
                cur, cur_kb = [], 0.0
            cur.append((mode, msha, path))
            cur_kb += kb
        if cur:
            batches.append(cur)

        for batch in batches:
            tree = self._build_tree_on(prev, batch)
            seed_env = dict(
                os.environ,
                GIT_AUTHOR_NAME="gitee-mirror",
                GIT_AUTHOR_EMAIL="mirror@localhost",
                GIT_COMMITTER_NAME="gitee-mirror",
                GIT_COMMITTER_EMAIL="mirror@localhost",
                GIT_AUTHOR_DATE=env_meta.get("GIT_AUTHOR_DATE", "@0 +0000"),
                GIT_COMMITTER_DATE=env_meta.get("GIT_AUTHOR_DATE", "@0 +0000"),
            )
            seed = self._commit_tree(tree, prev, b"blob seed batch\n", env_meta=seed_env)
            self.push_seq.append(seed)
            prev = seed

        new_tree = self.rebuild_tree(orig_tree)
        msg = raw.split(b"\n\n", 1)[1] if b"\n\n" in raw else b"mirrored\n"
        m = self._commit_tree(new_tree, prev, msg, env_meta=env_meta)
        self.commit_map[orig] = m
        self.push_seq.append(m)
        print(
            f"  {orig[:8]} -> {m[:8]}"
            f" (+{len(batches)} seeds, {sum(k for *_, k in entries):.0f}KB new blobs)",
            flush=True,
        )
        return m


def pack_kb(want: str, have: list[str]) -> float:
    revs = (want + "\n" + "".join("^" + h + "\n" for h in have)).encode()
    r = subprocess.run(
        ["git", "-C", MIRROR_DIR, "pack-objects", "--revs", "--stdout"],
        input=revs,
        capture_output=True,
    )
    return len(r.stdout) / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    init_mirror()
    print("[gitee-mirror] scanning for untransportable blobs ...", flush=True)
    repl = scan_replacements()
    chain = MirrorChain(repl)

    commits = gg("rev-list", "--reverse", "--topo-order", "HEAD").decode().split()
    print(f"[gitee-mirror] mirroring {len(commits)} commits ...", flush=True)
    for c in commits:
        m = chain.mirror_commit(c)
    head_mirror = chain.commit_map[commits[-1]]

    # record mapping for incremental reuse
    with open(os.path.join(MIRROR_DIR, ".git", "MIRROR_HEAD"), "w") as f:
        f.write(head_mirror + "\n")

    if args.no_push:
        print(f"[gitee-mirror] mirror head: {head_mirror[:12]} (push skipped)")
        return

    # remote refs already present count as "have" for thin-pack negotiation
    remote_refs = subprocess.run(
        ["git", "-C", MIRROR_DIR, "ls-remote", REMOTE_NAME],
        capture_output=True, text=True,
    ).stdout
    remote_known = [l.split()[0] for l in remote_refs.strip().splitlines() if l.strip()]
    remote_ref_names = [
        l.split()[1] for l in remote_refs.strip().splitlines() if l.strip()
    ]

    # stale _sm* refs from earlier seeding experiments collide with the new
    # chain ref names (non-fast-forward) - delete them first
    for name in remote_ref_names:
        short = name.split("/")[-1]
        if short.startswith("_sm") and short not in ("main",):
            ok, out = push(f":{name}")
            if not ok and "remote ref does not exist" not in out:
                print(f"  stale cleanup {short}: {out[:120]}", flush=True)

    n = 0
    for sha in chain.push_seq:
        if sha in remote_known:
            continue
        n += 1
        kb = pack_kb(sha, remote_known)
        if kb > PACK_BUDGET_KB:
            die(
                f"chain element {sha[:8]} pack {kb:.0f}KB exceeds budget "
                f"{PACK_BUDGET_KB:.0f}KB (seed split bug)"
            )
        ok, out = push(f"{sha}:refs/heads/_sm{n}")
        if not ok:
            die(f"push chain element {sha[:8]} failed ({kb:.0f}KB):\n{out}")
        remote_known.append(sha)
        print(f"  chain[{n}] {sha[:8]} ok ({kb:.0f}KB)")

    ok, out = push(f"{head_mirror}:refs/heads/main")
    print("[gitee-mirror] main push:", "ok" if ok else f"FAILED\n{out}")
    if not ok:
        sys.exit(1)

    # cleanup: all temporary branches (current seeds + leftover experiment refs)
    to_delete = [r.split("/")[-1] for r in remote_ref_names]
    to_delete += [f"_sm{i}" for i in range(1, n + 1)]
    to_delete += [
        r for r in ("_tmpbatch1", "_tmpbatch2", "_tmpbatch3a",
                    "_probe_fig4", "_probe_fig2", "_probe_fig1", "_probe_uvlock")
        if any(r in name for name in remote_ref_names)
    ]
    for ref in sorted(set(to_delete)):
        if not ref.startswith(("_sm", "_tmpbatch", "_probe")):
            continue
        ok, out = push(f":refs/heads/{ref}")
        if not ok and "remote ref does not exist" not in out:
            print(f"  cleanup {ref}: {out[:120]}")
    print("[gitee-mirror] DONE. Remember to set the gitee default branch to 'main'")


if __name__ == "__main__":
    main()
