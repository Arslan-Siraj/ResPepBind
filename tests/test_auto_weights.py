#!/usr/bin/env python3
"""Offline test of the weights installer and the automatic first-use cache."""
import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from respepbind import weights


def main():
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        archive = base / weights.ASSET
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in range(1, 11):
                z.writestr(f'outer/models/seed2026/fold{n}/weights.pt', f'fake checkpoint {n}')
            z.writestr('outer/models/seed2026/ensemble_independent_test_metrics.csv',
                       'ensemble_threshold_strategy,threshold\noof_youden,0.4309651851654053\n')
        no_report_archive = base / 'no_report.zip'
        with zipfile.ZipFile(no_report_archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in range(1, 11):
                z.writestr(f'outer/fold{n}/model.pt', f'fake checkpoint {n}')
        fallback = weights.install(base / 'fallback' / 'seed2026', archive_path=no_report_archive)
        assert weights.is_installed(fallback)
        assert '0.4309651851654053' in (fallback / 'ensemble_independent_test_metrics.csv').read_text()
        print('PASS: fold-only GitHub archive uses pinned training-derived threshold')
        installed = weights.install(base / 'cache' / 'seed2026', archive_path=archive)
        assert len(weights._fold_files(installed)) == 10
        assert weights.is_installed(installed)
        # Simulate automatic default first-use setup, without network.
        orig_default = weights.DEFAULT_MODEL_DIR
        orig_install = weights.install
        called = []
        try:
            weights.DEFAULT_MODEL_DIR = base / 'autocache' / 'seed2026'
            def fake_installer(destination, **options):
                called.append(str(destination))
                return orig_install(destination, archive_path=archive)
            weights.install = fake_installer
            auto = weights.ensure_installed(weights.DEFAULT_MODEL_DIR)
            assert auto.is_dir() and len(called) == 1
            auto2 = weights.ensure_installed(weights.DEFAULT_MODEL_DIR)
            assert auto2 == auto and len(called) == 1
            print('PASS: explicit install, first-run auto-install and reuse of all ten folds')
        finally:
            weights.DEFAULT_MODEL_DIR = orig_default
            weights.install = orig_install
        try:
            weights.ensure_installed(base / 'missing' / 'model', allow_download=False)
        except FileNotFoundError:
            print('PASS: offline/no-download policy')
        else:
            raise AssertionError('Did not refuse missing weights in offline mode')

if __name__ == '__main__':
    main()
