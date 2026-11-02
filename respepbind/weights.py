#!/usr/bin/env python3
"""Fetch and cache the official ResPepBind seed-2026 ten-fold model release.

Only standard-library modules are used here, so checkpoint download does not
require ProtT5, torch, or any numerical libraries. Model checkpoints are
subsequently read with torch.load(weights_only=True) by the predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_TAG = 'ResPepBind_model_v1.0.0_seed2026_10fold'
ASSET = 'ResPepBind_model_v1.0.0_seed2026_10fold.zip'
RELEASE_URL = (
    'https://github.com/Arslan-Siraj/ResPepBind/releases/download/'
    + RELEASE_TAG + '/' + ASSET
)
DEFAULT_MODEL_DIR = Path.home() / '.cache' / 'respepbind' / 'models' / 'seed2026'
DEFAULT_THRESHOLD = 0.4309651851654053
MAX_ARCHIVE_BYTES = 150_000_000  # Current public release is about 25.6 MB.
MAX_MEMBER_BYTES = 150_000_000


def _fold_files(directory: Path) -> list[Path]:
    paths = []
    for n in range(1, 11):
        folder = directory / f'fold{n}'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing {folder}')
        weights = list(folder.glob('*.pt'))
        if len(weights) != 1:
            raise ValueError(f'Expected exactly one .pt file in {folder}, found {len(weights)}')
        paths.append(weights[0])
    if not (directory / 'ensemble_independent_test_metrics.csv').is_file():
        raise FileNotFoundError(f'Missing threshold report in {directory}')
    return paths


def is_installed(directory: Path) -> bool:
    try:
        _fold_files(directory)
        return True
    except (FileNotFoundError, ValueError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _download(path: Path) -> None:
    print(f'[WEIGHTS] Downloading {RELEASE_URL}', flush=True)
    request = urllib.request.Request(RELEASE_URL, headers={'User-Agent': 'ResPepBind/1.0.0'})
    with urllib.request.urlopen(request, timeout=90) as response, path.open('wb') as dst:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError('Refusing abnormally large checkpoint archive')
            dst.write(chunk)
    print(f'[WEIGHTS] Downloaded {total:,} bytes', flush=True)


def _find_members(archive: zipfile.ZipFile):
    folds = {}
    reports = []
    for member in archive.infolist():
        if member.is_dir():
            continue
        if member.file_size > MAX_MEMBER_BYTES:
            raise ValueError('Oversized archive member: ' + member.filename)
        # Do not trust ZIP entry names: we never call extractall().
        raw = member.filename.replace('\\', '/')
        parts = raw.split('/')
        if raw.startswith('/') or any(part in ('', '..', '.') for part in parts):
            raise ValueError('Unsafe archive entry: ' + member.filename)
        mode = (member.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError('Symlink not allowed in model ZIP: ' + member.filename)
        if parts[-1] == 'ensemble_independent_test_metrics.csv':
            reports.append(member)
        elif parts[-1].lower().endswith('.pt'):
            dirs = [part for part in parts[:-1] if re.fullmatch('fold(?:[1-9]|10)', part, flags=re.I)]
            if len(dirs) == 1:
                idx = int(dirs[0][4:])
                if idx in folds:
                    raise ValueError(f'Found multiple checkpoint files for fold{idx}')
                folds[idx] = member
    if set(folds) != set(range(1, 11)):
        raise ValueError('Archive must contain exactly one .pt file in each fold1..fold10 directory; '
                         f'found fold numbers: {sorted(folds)}')
    if reports:
        reports.sort(key=lambda item: len(item.filename.split('/')))
    return folds, reports[0] if reports else None


def _threshold(report_bytes: bytes) -> float:
    reader = csv.DictReader(io.StringIO(report_bytes.decode('utf-8-sig')))
    values = {float(row['threshold']) for row in reader
              if row.get('ensemble_threshold_strategy') == 'oof_youden'}
    if len(values) != 1:
        raise ValueError('Expected a single OOF Youden threshold in report')
    value = values.pop()
    if abs(value - DEFAULT_THRESHOLD) > 1e-9:
        raise ValueError(f'Released threshold {value} does not match verified {DEFAULT_THRESHOLD}')
    return value


def install(destination: Path = DEFAULT_MODEL_DIR, *, archive_path: Path | None = None,
            force: bool = False, expected_sha256: str | None = None) -> Path:
    destination = Path(destination).expanduser().resolve()
    if is_installed(destination) and not force:
        print('[WEIGHTS] Using installed ten-fold ensemble:', destination, flush=True)
        return destination
    if destination.exists() and not force:
        raise RuntimeError(f'Model cache exists but is incomplete: {destination}. '
                           'Use `respepbind-download-weights --force` to replace it.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='respepbind-weights-', dir=str(destination.parent)) as temp_dir:
        temp = Path(temp_dir)
        archive = Path(archive_path) if archive_path is not None else temp / ASSET
        if archive_path is None:
            _download(archive)
        if not archive.is_file():
            raise FileNotFoundError(archive)
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError('Checkpoint archive exceeds expected maximum size')
        digest = sha256_file(archive)
        print('[WEIGHTS] SHA256:', digest, flush=True)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError('Checkpoint ZIP SHA256 mismatch')
        staging = temp / 'seed2026'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise ValueError('Corrupted checkpoint ZIP')
            members, report = _find_members(z)
            if report is None:
                # The trained fold archive may contain only .pt files.
                # Use the fixed, independently verified training-derived threshold.
                report_bytes = (
                    'ensemble_threshold_strategy,threshold\n'
                    f'oof_youden,{DEFAULT_THRESHOLD}\n'
                ).encode('utf-8')
                threshold_source = 'verified_repo_training_report_v1.0.0'
                print('[WEIGHTS] Release has no threshold CSV; using the fixed verified training-derived threshold', flush=True)
            else:
                report_bytes = z.read(report)
                threshold_source = 'model_release_report'
            selected_threshold = _threshold(report_bytes)
            details = {}
            for idx in range(1, 11):
                member = members[idx]
                folder = staging / f'fold{idx}'
                folder.mkdir()
                target = folder / Path(member.filename).name
                with z.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                details[f'fold{idx}'] = {'file': target.name, 'bytes': target.stat().st_size}
            (staging / 'ensemble_independent_test_metrics.csv').write_bytes(report_bytes)
            (staging / 'download_metadata.json').write_text(json.dumps({
                'release_tag': RELEASE_TAG, 'source_url': RELEASE_URL if archive_path is None else str(archive),
                'sha256': digest, 'threshold': selected_threshold, 'threshold_source': threshold_source, 'folds': details,
            }, indent=2) + '\n')
        if destination.exists():
            # The caller explicitly requested --force; completed extraction is staged first.
            shutil.rmtree(destination)
        staging.replace(destination)
    _fold_files(destination)
    print('[WEIGHTS] Ready:', destination, flush=True)
    return destination


def ensure_installed(destination: Path = DEFAULT_MODEL_DIR, *, allow_download: bool = True) -> Path:
    destination = Path(destination).expanduser()
    if is_installed(destination):
        return destination
    if not allow_download:
        raise FileNotFoundError(f'No complete ten-fold model at {destination}. '
                                'Download weights or omit --no-download-weights.')
    return install(destination)


def main():
    parser = argparse.ArgumentParser(description='Install the official ResPepBind ten-fold weights')
    parser.add_argument('--dest', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--archive', type=Path, help='Use an existing downloaded ZIP instead of the network')
    parser.add_argument('--expected-sha256', help='Expected release ZIP SHA256 (optional)')
    parser.add_argument('--force', action='store_true', help='Replace an incomplete or existing install')
    args = parser.parse_args()
    install(args.dest, archive_path=args.archive, force=args.force,
            expected_sha256=args.expected_sha256)


if __name__ == '__main__':
    main()
