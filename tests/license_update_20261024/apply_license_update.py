"""Apply the approved license organization without changing executable code."""

from pathlib import Path
import ast
import hashlib
import json
import shutil
import tempfile
import tomllib


ROOT = Path('/diancpfs/user/wanghanzhen/LiteSpec_publish')
STAGE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    original_license = (ROOT / 'LICENSE').read_bytes()
    assert original_license == (STAGE / 'SpecForge-MIT.txt').read_bytes()
    assert not (ROOT / 'THIRD_PARTY_NOTICES.md').exists()
    assert not (ROOT / 'licenses').exists()
    project = (ROOT / 'pyproject.toml').read_text()
    assert tomllib.loads(project)['project']['license'] == 'Apache-2.0'
    assert 'license-files = ["LICENSE"]' in project
    project = project.replace('license-files = ["LICENSE"]',
                              'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt"]', 1)
    readme = (ROOT / 'README.md').read_text()
    old = '## License\n\nSee [LICENSE](LICENSE).'
    assert old in readme
    readme = readme.replace(old, '''## License

LiteSpec is distributed under the [Apache License, Version 2.0](LICENSE).
Third-party portions retain their original licenses and copyright notices,
including SpecForge's MIT-licensed code. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution and
the [licenses](licenses/) directory for the preserved upstream license texts.''', 1)
    manifest = (ROOT / 'MANIFEST.in').read_text()
    assert 'include THIRD_PARTY_NOTICES.md' not in manifest
    manifest = ('include LICENSE THIRD_PARTY_NOTICES.md\n'
                'recursive-include licenses *.txt\n' + manifest)
    pending = {'LICENSE': (STAGE / 'Apache-2.0.txt').read_bytes(),
               'THIRD_PARTY_NOTICES.md': (STAGE / 'THIRD_PARTY_NOTICES.md').read_bytes(),
               'pyproject.toml': project.encode(), 'README.md': readme.encode(),
               'MANIFEST.in': manifest.encode()}
    for path in STAGE.glob('*.txt'):
        if path.name != 'Apache-2.0.txt':
            pending['licenses/' + path.name] = (original_license if path.name == 'SpecForge-MIT.txt'
                                               else path.read_bytes())

    descriptions = {
        'litespec/data/preprocessing.py': 'integrate data preparation with LiteSpec training and capture.',
        'litespec/data/utils.py': 'integrate data processing with the LiteSpec training framework.',
        'litespec/data/template.py': 'integrate chat-template registration with LiteSpec data preparation.',
        'litespec/modeling/draft/flex_attention.py': 'integrate flexible attention with LiteSpec draft models.',
        'litespec/modeling/target/sglang_backend/model_runner.py': 'bind SGLang runner execution to LiteSpec distributed groups.',
        'litespec/modeling/target/sglang_backend/patch.py': 'integrate SGLang initialization with LiteSpec process groups.',
        'litespec/modeling/target/sglang_backend/utils.py': 'return hidden states through the LiteSpec target interface.',
        'litespec/distributed/collectives.py': 'integrate collective operations with LiteSpec process groups.',
        'litespec/layers/ring/__init__.py': 'integrate ring attention with LiteSpec attention interfaces.',
        'litespec/layers/ring/ring_flash_attn.py': 'integrate ring attention with LiteSpec attention interfaces.',
        'litespec/layers/ring/utils.py': 'integrate ring-attention communication with LiteSpec execution.',
    }
    custom = ROOT / 'litespec/modeling/target/custom_backend'
    for name in ['gpt_oss.py', 'llama.py', 'llama4.py', 'phi3.py', 'qwen2.py', 'qwen3.py', 'qwen3_moe.py']:
        descriptions[str((custom / name).relative_to(ROOT))] = 'adapt target execution and hidden-state access to LiteSpec.'
    for name, description in descriptions.items():
        before = (ROOT / name).read_bytes()
        bom = b'\xef\xbb\xbf' if before.startswith(b'\xef\xbb\xbf') else b''
        content = before[len(bom):].decode()
        notice = ('# Modified for LiteSpec: ' + description + '\n'
                  '# Upstream attribution and license terms: THIRD_PARTY_NOTICES.md.\n')
        if content.startswith('# coding='):
            first, rest = content.split('\n', 1)
            after = first + '\n' + notice + rest
        else:
            after = notice + '\n' + content
        updated = bom + after.encode()
        assert ast.dump(ast.parse(before), include_attributes=False) == ast.dump(ast.parse(updated), include_attributes=False)
        pending[name] = updated

    backup = Path(tempfile.mkdtemp(prefix='license_backup_', dir=STAGE))
    originals = {}
    for name in pending:
        source = ROOT / name
        if source.exists():
            original = source.read_bytes()
            dest = backup / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(original)
            originals[name] = digest(original)
        else:
            originals[name] = None
    for name, data in pending.items():
        target = ROOT / name
        assert (digest(target.read_bytes()) if target.exists() else None) == originals[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.' + target.name + '.license-update.tmp')
        temporary.write_bytes(data)
        temporary.replace(target)
    assert (ROOT / 'licenses/SpecForge-MIT.txt').read_bytes() == original_license
    receipt = {'project_license': 'Apache-2.0', 'specforge_mit_preserved_byte_for_byte': True,
               'python_changes': 'comments only; all modified ASTs identical',
               'modified_python_files': len(descriptions), 'original_hashes': originals,
               'new_hashes': {name: digest(data) for name, data in pending.items()}}
    (STAGE / 'application_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if not key.endswith('hashes')}, indent=2))


if __name__ == '__main__':
    main()
