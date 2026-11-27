"""Stage, test and activate Hugging Face H3 weights for an existing installation."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
from tempfile import TemporaryDirectory
import time
from urllib.parse import quote

from . import disk_space
from .adaln_assets import asset_path, file_hash
from .hf_model_jobs import HubModelJobs, public_candidate
from .long_paths import io_path
from .monitoring import load_ledger, save
from .storage import fingerprint

ACTIVE_KEY = 'hf_video_model'
OFFICIAL_KEY = 'hf_video_official'
OFFICIAL_ID = 'official'
PROFILE_FILE = 'profile.json'
PROFILE_ID = re.compile('[0-9a-f]{24}')
SMOKE_TIMEOUT = 30 * 60
BASELINE_FIELDS = ('base', 'checkpoint', 'cache', 'prepared_format')
WORKER_MODULE = 'freevideo_engine.hf_video_models'


def profile_folder(root, identity):
    if not isinstance(identity, str) or not PROFILE_ID.fullmatch(identity):
        raise ValueError('Invalid video model ID.')
    return asset_path(Path(root) / 'prepared', 'huggingface/' + identity)


def _read(path):
    return json.loads(io_path(path).read_text(encoding='utf-8'))


def _baseline(machine):
    return {key: machine[key] for key in BASELINE_FIELDS if key in machine}


def _format(manifest):
    return 'int8_convrot' if manifest['precision'] == 'int8' else 'fp8_' + manifest['scale_granularity']


@lru_cache(maxsize=1)
def _official_commits():
    from .prepared_model import catalog
    spec = catalog()
    return {
        Path('edge-' + revision[:16]) / variant['cache_prefix']: (spec['repo'], revision)
        for variant in spec['variants'].values()
        for revision in (variant.get('revision', spec['revision']),)
    }


def _official(machine, root):
    baseline = machine.get(OFFICIAL_KEY) or _baseline(machine)
    cache = Path(baseline['cache'])
    value = dict(id=OFFICIAL_ID, label='FreeVideo', repo='OpenVDN/vdn-minimax-h3',
                 revision=machine.get('model_revision', ''), format=baseline.get('prepared_format', 'fp8'))
    # A prepared official package records an immutable Edge commit in its directory name.
    prefix = (root / 'prepared').resolve()
    resolved = cache.resolve()
    published = _official_commits().get(resolved.relative_to(prefix)) if resolved.is_relative_to(prefix) else None
    if published:
        value.update(repo=published[0], revision=published[1])
    # JSON file sizes remain tiny compared with the tensors. No tensor reads occur here.
    try:
        value['format'] = _format(_read(cache / 'manifest.json'))
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return value


def _contract(machine):
    return dict(
        transformer=file_hash(io_path(Path(machine['base']) / 'transformer' / 'config.json')),
        vdn=file_hash(io_path(Path(machine['checkpoint']) / 'model_spec.json')),
        device_backend=machine.get('device_backend', 'cuda'),
        gpu_uuid=machine.get('gpu_uuid') or machine.get('device_identity'),
    )


def _receipt(folder):
    value = _read(folder / PROFILE_FILE)
    if (value.get('schema_version') not in (1, 2) or value['model']['id'] != folder.name
            or not isinstance(value.get('stamps'), dict)):
        raise ValueError('Invalid installed video model receipt.')
    if value['schema_version'] == 2:
        if value['model'].get('backend') != 'comfy_h3' or value['model'].get('kind') != 'comfy_h3':
            raise ValueError('Invalid installed ComfyUI H3 model receipt.')
        return value, None
    cache = asset_path(folder, value['cache'])
    if not (cache / 'manifest.json').is_file():
        raise ValueError('The model files are missing. Import the model again.')
    return value, cache


def _comfy_contract(machine):
    return dict(recipe_version=1, device_backend=machine.get('device_backend', 'cuda'),
                gpu_uuid=machine.get('gpu_uuid') or machine.get('device_identity'))


def _comfy_profile(root, model, assets):
    from .local_comfy_h3_catalog import PRESETS, profile_assets
    if not isinstance(model.get('preset'), str) or model['preset'] not in PRESETS:
        raise ValueError('Invalid ComfyUI H3 workflow preset.')
    paths = profile_assets(root, assets)
    return dict(model, variant=model['preset'], assets=paths, **paths)


def _verified_comfy_profile(root, receipt, machine):
    from .local_comfy_h3_catalog import owned_stamp
    if receipt.get('contract') != _comfy_contract(machine):
        raise ValueError('This model was tested on a different FreeVideo configuration. Import it again.')
    profile = _comfy_profile(root, receipt['model'], receipt['assets'])
    if {role: owned_stamp(root, Path(value)) for role, value in profile['assets'].items()} != receipt['stamps']:
        raise ValueError('The installed video model changed. Inspect and import it again.')
    return profile


def _with_comfy_encoder(profile, encoder):
    if encoder is None:
        return profile
    path = str(encoder['path'])
    return dict(profile, encoder=path, assets=dict(profile['assets'], encoder=path),
                shared_encoder={key: encoder[key] for key in ('id', 'label', 'sha256')})


def _with_comfy_vaes(root, profile):
    from .model_components_generation import native_vae_files
    assets = native_vae_files(root, profile['assets'])
    return dict(profile, **assets, assets=assets)


def comfy_shared_encoder_profile(root, identity, machine, encoder):
    """Bind a validated shared component to a preset without editing its installed asset receipt."""
    folder = profile_folder(root, identity)
    receipt, cache = _receipt(folder)
    if cache is not None:
        raise ValueError('The selected video model is not a ComfyUI H3 checkpoint.')
    profile = _with_comfy_encoder(_verified_comfy_profile(root, receipt, machine), encoder)
    return _with_comfy_vaes(root, profile)


def resolve_comfy_h3(root, identity, machine=None, *, encoder_component=None):
    """Resolve a tested profile to private, owned files for the native ComfyUI loader."""
    from .encoder_checkpoint import ACTIVE_ENCODER, configured_component
    root = Path(root).resolve()
    machine = machine if machine is not None else _read(root / 'machine.json')
    encoder = configured_component(machine, ACTIVE_ENCODER if encoder_component is None else encoder_component)
    return comfy_shared_encoder_profile(root, identity, machine, encoder)


def active_comfy_h3(root, machine=None):
    """Return the verified active ComfyUI H3 profile, if one is selected."""
    root = Path(root)
    machine = machine if machine is not None else _read(root / 'machine.json')
    active = machine.get(ACTIVE_KEY) or {}
    if active.get('backend') == 'comfy_h3':
        return resolve_comfy_h3(root, active['id'], machine)
    return None


def smoke_shared_encoder(run_child, machine, folder, descriptor, identity, *, reference=False, phase='smoke-video'):
    """Verify the explicit shared encoder in the VDN installation under the caller's GPU lease."""
    from .encoder_checkpoint import BUILTIN_ENCODER
    from .managed import command_for

    identity = identity or BUILTIN_ENCODER
    attempt = folder / (phase + '-' + str(time.time_ns()))
    attempt.mkdir()
    prompt, output = attempt / 'prompt.txt', attempt / 'video.mp4'
    caption = 'A blue ceramic cup is still on a table in soft daylight. Quiet room tone.'
    args = ['--encoder-component', identity, '--prompt-file', str(prompt), '--out', str(output),
            '--width', '256', '--height', '256', '--frames', '39', '--no-two-pass', '--no-tuning']
    if reference:
        from PIL import Image
        picture = attempt / 'reference.png'
        Image.new('RGB', (64, 64), (95, 133, 168)).save(picture)
        media = attempt / 'media.json'
        save(media, {'version': 1, 'references': [{'kind': 'image', 'path': str(picture)}]})
        args += ['--media', str(media)]
        caption = '<Picture 1> is the visual reference. ' + caption
    prompt.write_text(caption + '\n', encoding='utf-8')
    command = command_for(machine, 'generate', args)
    run_child(machine, attempt, phase, command[1:], descriptor=descriptor, timeout=SMOKE_TIMEOUT)
    report = _read(output.with_suffix('.request.json'))
    if (not output.is_file() or output.stat().st_size < 32 or report.get('success') is not True
            or report.get('encoding', {}).get('text_encoder', {}).get('id') != identity):
        raise ValueError('The candidate H3 encoder did not complete the VDN test video.')
    run_child(machine, attempt, phase + '-validation', ['-m', WORKER_MODULE, 'validate-video', str(output)], timeout=30)


def _conversion_cache(folder, report):
    cache = asset_path(folder / 'converted', _read(report)['cache'])
    if not (cache / 'manifest.json').is_file():
        raise ValueError('The prepared video model cache is incomplete.')
    return cache


def _converted_bytes(folder):
    """Discount an existing conversion after a retry reached GPU validation."""
    try:
        cache = _conversion_cache(folder, folder / 'conversion.json')
        manifest = _read(cache / 'manifest.json')
        return sum(row['bytes'] for row in manifest['groups']
                   if asset_path(cache, row['file']).stat().st_size == row['bytes'])
    except (OSError, ValueError, TypeError, KeyError):
        return 0


def _stamp_cache(cache, manifest):
    from .adaln_assets import projection_groups, validate_catalog
    count = sum(row['group'].startswith('blocks/') for row in manifest['groups'])
    payload = {row['file'] for row in manifest['groups']}
    payload.update(row['file'] for row, _ in validate_catalog(manifest, count))
    payload.update(row['file'] for row in projection_groups(manifest)
                   if asset_path(cache, row['file']).is_file())
    return {name: fingerprint(asset_path(cache, name)) for name in sorted(payload | {'manifest.json'})}


def _retained_profile(identity, folder):
    from .model_components import retained_use
    try:
        from folder_paths import get_output_directory
        output = Path(get_output_directory())
    except (ImportError, AttributeError, OSError, TypeError):
        return False
    if retained_use(output, identity):
        return True
    root = (output / 'FreeVideo').resolve()
    folder = folder.resolve()
    for artifacts in root.glob('*/*/*.artifacts'):
        if artifacts.is_symlink() or not artifacts.resolve().is_relative_to(root):
            continue
        if not any((artifacts / name).is_file() for name in ('first-pass.pt', 'latents.pt', 'refine-input.pt', 'refine-progress.pt')):
            continue
        try:
            report = artifacts.with_suffix('.request.json')
            summary = _read(report) if report.stat().st_size <= 2 * 1024 * 1024 and not report.is_symlink() else {}
            if summary.get('success') is True and not summary.get('preview'):
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        for name in ('video.json', 'engine-request.json', 'lora-request.json'):
            path = artifacts / name
            try:
                if path.stat().st_size > 2 * 1024 * 1024 or path.is_symlink():
                    continue
                cache = _read(path).get('cache')
                if isinstance(cache, str) and Path(cache).resolve().is_relative_to(folder):
                    return True
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    return False


class VideoModels(HubModelJobs):
    media = 'video'

    def __init__(self, *args, comfy_smoke=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_smoke = comfy_smoke
        self.comfy_selections = {}

    def _machine(self):
        try:
            value = _read(self.root / 'machine.json')
            if value.get('ready') and value.get('video_model', True) and all(
                    value.get(key) for key in ('base', 'checkpoint', 'cache', 'python')):
                return value
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        raise ValueError('Install the FreeVideo H3 video model before importing another model.')

    def status(self, *, active_only=False):
        with self.mutex:
            task = dict(self.task)
        try:
            machine = self._machine()
        except ValueError:
            return dict(available=False, active=None, installed=[], task=task)
        builtin = dict(_official(machine, self.root), repairable=False, removable=False)
        installed, damaged = [builtin], []
        active = machine.get(ACTIVE_KEY) or builtin
        profiles = self.root / 'prepared' / 'huggingface'
        identity = active.get('id')
        if active_only:
            directories = (profiles / identity,) if isinstance(identity, str) and PROFILE_ID.fullmatch(identity) else ()
        else:
            directories = sorted(profiles.glob('*'))
        for directory in directories:
            if not PROFILE_ID.fullmatch(directory.name):
                continue
            row = None
            try:
                folder = profile_folder(self.root, directory.name)
                receipt = self._maintenance_record(folder)
                row = dict(receipt['model'], repairable=self._repairable(receipt), removable=directory.name != active['id'])
                _, cache = _receipt(folder)
                if cache is None:
                    _verified_comfy_profile(self.root, receipt, machine)
                installed.append(row)
            except (OSError, ValueError, TypeError, KeyError):
                if row is not None:
                    damaged.append(dict(row, damaged=True))
        current = next((row for row in (*installed, *damaged) if row['id'] == active['id']), None)
        if current is None:
            current = dict(active, damaged=True, repairable=False, removable=False)
            damaged.append(current)
        return dict(available=True, active=current, installed=installed, damaged=damaged, task=task)

    def _maintenance_record(self, folder):
        receipt = _read(folder / PROFILE_FILE)
        if receipt.get('schema_version') not in (1, 2) or receipt['model']['id'] != folder.name:
            raise ValueError('Invalid installed video model receipt.')
        return receipt

    @staticmethod
    def _repairable(receipt):
        model = receipt['model']
        return bool(receipt.get('origin') or model.get('repo') and model.get('revision'))

    def verify(self, identity):
        folder = profile_folder(self.root, identity) if identity != OFFICIAL_ID else None
        return self._maintenance(identity, 'verify', self._verify_installation, folder)

    def _verify_installation(self, folder):
        from . import provision
        machine = self._machine()
        if folder is None:
            baseline = machine.get(OFFICIAL_KEY) or _baseline(machine)
            cache = Path(baseline['cache'])
        else:
            receipt, cache = _receipt(folder)
            if cache is None:
                profile = _verified_comfy_profile(self.root, receipt, machine)
                for path in profile['assets'].values():
                    self._check()
                    path = Path(path)
                    if file_hash(io_path(path)) != path.parent.name:
                        raise ValueError('A native H3 asset failed SHA-256 verification.')
                return
        manifest = provision.verify_cache(dict(root=str(folder or self.root), model_dir=str(Path(machine['base']).parent),
                                               verification='full'), cache)
        if folder is not None and _stamp_cache(cache, manifest) != receipt['stamps']:
            raise ValueError('The installed video model files changed. Repair or import it again.')

    def remove(self, identity):
        folder = profile_folder(self.root, identity)
        self._maintenance_record(folder)
        return self._maintenance(identity, 'remove', self._remove_version, identity, folder)

    def _remove_version(self, identity, folder):
        with self._maintenance_lease():
            if (self._machine().get(ACTIVE_KEY) or {}).get('id') == identity:
                raise ValueError('Activate another video model before removing this version.')
            receipt = self._maintenance_record(folder)
            if _retained_profile(identity, folder):
                raise ValueError('A saved preview or recoverable video still uses this model.')
            self._delete_import(folder)
            if receipt['model'].get('kind') == 'comfy_h3':
                self._remove_native_checkpoint(receipt)

    def _remove_native_checkpoint(self, receipt):
        from .local_comfy_h3_catalog import owned_asset, OWNED, _asset_receipt
        relative = receipt['assets'].get('checkpoint')
        if not isinstance(relative, str):
            return
        path = asset_path(self.root, relative)
        if path != owned_asset(self.root, 'checkpoint', path.parent.name):
            return
        for folder in sorted((self.root / 'prepared' / 'huggingface').glob('*')):
            if not PROFILE_ID.fullmatch(folder.name):
                continue
            try:
                if self._maintenance_record(folder).get('assets', {}).get('checkpoint') == relative:
                    return
            except (OSError, ValueError, TypeError, KeyError):
                return  # A damaged receipt can still own the checkpoint.
        if path.parent.parent == asset_path(self.root, OWNED):
            io_path(path).unlink(missing_ok=True)
            io_path(_asset_receipt(path)).unlink(missing_ok=True)

    def repair(self, identity, *, token=None):
        from .hf_auth import validate
        folder = profile_folder(self.root, identity)
        receipt = self._maintenance_record(folder)
        if not self._repairable(receipt):
            raise ValueError('Inspect the original model again before replacing this version.')
        token = validate(token or '')
        return self._maintenance(identity, 'repair', self._repair_version, identity, folder, receipt, token, token=token)

    def _repair_version(self, identity, folder, receipt, token):
        from . import hf_video_catalog, local_comfy_h3_catalog, local_video_catalog
        machine = self._machine()
        model, origin = receipt['model'], receipt.get('origin') or {}
        source = origin.get('path')
        local = source is not None
        if model.get('kind') == 'comfy_h3':
            self._require_native()
            if local:
                candidate = local_comfy_h3_catalog.inspect_file(source, model['preset'], self.root, machine)
            else:
                url = 'https://huggingface.co/' + model['repo'] + '/blob/' + quote(model['revision'], safe='') + '/' + quote(model['path'])
                candidate = local_comfy_h3_catalog.inspect_hub(url, model['preset'], self.root, machine, token)
        elif local:
            candidate = local_video_catalog.inspect(source, self.root, machine)
        else:
            candidate = hf_video_catalog.inspect(model['repo'], self.root, machine, revision=model['revision'],
                                                subfolder=model.get('subfolder'), token=token)
        if candidate.get('compatible') is False or not local and candidate['id'] != identity:
            raise ValueError('The original source no longer matches this video model.')
        candidate = dict(candidate, id=identity, source_selection=source)
        with TemporaryDirectory(prefix='repair-video-', dir=io_path(folder.parent)) as temporary:
            staged = Path(temporary) / identity
            staged.mkdir()
            if local:
                self._update('copying', total_bytes=candidate['copy_bytes'])
                hashes = self._copy_local_files(staged, candidate, lambda done: self._update('copying', done_bytes=done))
                if hashes != [tuple(row) for row in origin.get('hashes', ())]:
                    raise ValueError('The original local model changed. Import it as a new version.')
                candidate['local_hashes'] = hashes
            self._stage(candidate, token, folder=staged, remote=not local, repair=True)
            with self._maintenance_lease():
                self._replace_import(folder, staged, lambda: self._refresh_repaired_video(identity, folder))

    def _refresh_repaired_video(self, identity, folder):
        machine = self._machine()
        receipt, cache = _receipt(folder)
        if cache is None:
            self._refresh_native_neighbors(folder, receipt)
        if (machine.get(ACTIVE_KEY) or {}).get('id') != identity:
            return
        if cache is not None:
            manifest = _read(cache / 'manifest.json')
            machine.update(cache=str(cache), prepared_format='int8_convrot' if manifest['precision'] == 'int8' else 'fp8')
        machine[ACTIVE_KEY] = receipt['model']
        save(self.root / 'machine.json', machine, durable=True)

    def _refresh_native_neighbors(self, repaired, receipt):
        from .local_comfy_h3_catalog import owned_stamp, profile_assets
        shared = set(receipt['assets'].values())
        for folder in sorted(repaired.parent.glob('*')):
            if folder == repaired or not PROFILE_ID.fullmatch(folder.name):
                continue
            try:
                installed = self._maintenance_record(folder)
                if installed['schema_version'] != 2 or not shared.intersection(installed['assets'].values()):
                    continue
                paths = profile_assets(self.root, installed['assets'])
                stamps = {role: owned_stamp(self.root, Path(path)) for role, path in paths.items()}
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if stamps != installed['stamps']:
                save(folder / PROFILE_FILE, dict(installed, stamps=stamps), durable=True)

    def search(self, query, token=None):
        from . import hf_video_catalog
        self._machine()
        return hf_video_catalog.search(query, token=token)

    def inspect(self, reference, revision=None, subfolder=None, token=None, *, preset=None):
        from . import hf_video_catalog
        machine = self._machine()
        basename = reference.rsplit('/', 1)[-1].casefold() if isinstance(reference, str) else ''
        if preset is None and basename.endswith('.safetensors'):
            preset = next((name for name in ('fl2va', 'ref2va') if name in basename), None)
        if preset is None:
            candidate = hf_video_catalog.inspect(reference, self.root, machine,
                                                 revision=revision, subfolder=subfolder, token=token)
        else:
            self._require_native()
            from .hf_auth import validate
            from .local_comfy_h3_catalog import inspect_hub
            candidate = inspect_hub(reference, preset, self.root, machine, validate(token or ''))
        profile_folder(self.root, candidate['id'])
        return self._remember(candidate)

    def inspect_local(self, selection):
        from . import local_video_catalog
        candidate = local_video_catalog.inspect(selection, self.root, self._machine())
        return self._remember_local(dict(candidate, source_selection=str(selection)))

    def scan_comfy_h3(self, local_folder):
        from . import local_comfy_h3_catalog as catalog
        self._machine()
        selections = {secrets.token_hex(12): row for row in catalog.scan(local_folder)}
        with self.mutex:
            self.comfy_selections = selections
        return dict(candidates=[dict(public_candidate(row), id=identity, presets=row['presets'])
                                for identity, row in selections.items()])

    def inspect_comfy_h3(self, selection_id, preset):
        from . import local_comfy_h3_catalog as catalog
        with self.mutex:
            selection = self.comfy_selections.get(selection_id) if isinstance(selection_id, str) else None
        if selection is None:
            raise ValueError('Choose the ComfyUI H3 directory again.')
        self._require_native()
        candidate = catalog.inspect(selection, preset, self.root, self._machine())
        return self._remember_local(dict(candidate, source_selection=selection['files'][0]['source']))

    def _require_native(self):
        if self.comfy_smoke is None or not getattr(self.comfy_smoke, 'available', lambda: True)():
            raise ValueError('The current ComfyUI runtime cannot validate this H3 model.')

    def _profile_folder(self, identity):
        return profile_folder(self.root, identity)

    def _complete_profile(self, folder, machine):
        try:
            previous, cache = _receipt(folder)
            if cache is None:
                return public_candidate(_verified_comfy_profile(self.root, previous, machine))
            if (previous.get('contract') == _contract(machine)
                    and _stamp_cache(cache, _read(cache / 'manifest.json')) == previous['stamps']):
                return previous['model']
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        return None

    def _stage(self, candidate, token, *, folder=None, remote=False, repair=False):
        from . import hf_video_catalog, local_video_catalog, provision
        machine = self._machine()
        if candidate['kind'] == 'comfy_h3':
            return self._stage_comfy(machine, candidate, folder, token=token, remote=remote or folder is None, repair=repair)
        local = folder is not None and not remote
        direct = folder is None
        catalog = local_video_catalog if local else hf_video_catalog
        if not local:
            folder = folder or profile_folder(self.root, candidate['id'])
            folder.mkdir(parents=True, exist_ok=True)
            if self._complete_profile(folder, machine) is not None:
                self._update('ready', done_bytes=candidate['bytes'])
                return
            remaining, ledger = self._remaining(folder, candidate['files'])
            reserve = max(0, candidate.get('required_bytes', candidate['bytes']) -
                          candidate.get('download_bytes', candidate['bytes']))
            if candidate['kind'] == 'raw' and remaining == 0:
                reserve = max(0, reserve - _converted_bytes(folder))
            if disk_space.free_bytes(folder) < remaining + reserve:
                raise ValueError('Insufficient disk space for this video model.')
            self._download(machine, folder, candidate, token, ledger)
        if candidate['kind'] == 'raw':
            self._update('verifying')
            catalog.validate_import(candidate, folder, machine)
            conversion = folder / 'conversion.json'
            self._update('waiting_for_gpu')
            with self._smoke_lease() as descriptor:
                self._update('preparing')
                self._child(machine, folder, 'preparation',
                            ['-m', WORKER_MODULE, 'prepare', str(asset_path(folder, candidate['base'])),
                             str(asset_path(folder, candidate['checkpoint'])), str(folder / 'converted'), str(conversion)],
                            descriptor=descriptor)
            cache = _conversion_cache(folder, conversion)
        else:
            cache = asset_path(folder, candidate['cache'])
        self._check()
        self._update('verifying')
        catalog.validate_import(candidate, folder, machine, cache=cache)
        manifest = provision.verify_cache(dict(root=str(folder), model_dir=str(Path(machine['base']).parent)), cache)
        self._check()
        self._update('waiting_for_gpu')
        with self._smoke_lease() as descriptor:
            self._update('smoke')
            self._smoke(machine, folder, cache, descriptor)
        self._check()
        model = dict(public_candidate(candidate), format=_format(manifest))
        origin = dict(path=candidate['source_selection'], hashes=candidate['local_hashes']) if candidate.get('source_selection') else None
        save(folder / PROFILE_FILE, dict(schema_version=1, model=model, cache=cache.relative_to(folder).as_posix(),
                                        contract=_contract(machine), stamps=_stamp_cache(cache, manifest), origin=origin), durable=True)
        if direct:
            self._update('ready')
        return model

    def _copy_local_files(self, staging, selection, progress):
        if selection['kind'] == 'comfy_h3' and selection.get('reused_checkpoint'):
            from .local_comfy_h3_catalog import reused_checkpoint
            from .local_model_copy import source_record
            selected = selection['files'][0]
            actual = source_record(selected['source'], selected['target'], root=selected['root'])
            checksum = reused_checkpoint(self.root, actual)
            if actual['fingerprint'] != selected['fingerprint'] or checksum != selection['reused_checkpoint']:
                raise ValueError('The checkpoint changed after inspection. Select it again.')
            progress(0)
            return [(selected['target'], selected['bytes'], checksum)]
        return super()._copy_local_files(staging, selection, progress)

    def _stage_comfy(self, machine, candidate, folder, *, token='', remote=False, repair=False):
        from . import local_comfy_h3_catalog as catalog
        from .encoder_checkpoint import configured_component
        self._require_native()
        direct = folder is None
        if remote:
            folder = folder or self._profile_folder(candidate['id'])
            folder.mkdir(parents=True, exist_ok=True)
            if direct and self._complete_profile(folder, machine) is not None:
                self._update('ready', done_bytes=candidate['bytes'])
                return
            downloads = list(candidate['files'])
            if catalog._verified(self.root, 'checkpoint', downloads[0]['sha256']):
                downloads = downloads[1:]
            remaining, ledger = self._remaining(folder, downloads)
            if disk_space.free_bytes(folder) < remaining + candidate['copy_bytes'] + catalog.MARGIN:
                raise ValueError('Insufficient disk space for this native H3 model.')
            if downloads:
                self._download(machine, folder, dict(candidate, files=downloads), token, ledger)

        def progress(done):
            self._check()
            self._update('copying', done_bytes=done + (candidate['bytes'] if remote else 0),
                         total_bytes=(candidate['bytes'] + candidate['copy_bytes']) if remote else candidate['copy_bytes'])

        stage = catalog.stage_hub if remote else catalog.stage
        if repair:
            with self._maintenance_lease():
                assets = stage(self.root, folder, candidate, progress, repair=True)
        else:
            assets = stage(self.root, folder, candidate, progress)
        model = public_candidate(candidate)
        profile = _with_comfy_encoder(_comfy_profile(self.root, model, assets), configured_component(machine))
        profile = _with_comfy_vaes(self.root, profile)
        self._check()
        self._update('waiting_for_gpu')
        output = folder / 'comfy-h3-smoke.mp4'
        self._update('smoke')
        result = Path(self.comfy_smoke(profile, output, self._check))
        self._check()
        if not io_path(result).is_file() or io_path(result).stat().st_size < 32:
            raise ValueError('The candidate model did not produce a playable video.')
        self._child(machine, folder, 'smoke-validation', ['-m', WORKER_MODULE, 'validate-video', str(result)],
                    timeout=30)
        self._check()
        stamps = {role: catalog.owned_stamp(self.root, Path(path))
                  for role, path in catalog.profile_assets(self.root, assets).items()}
        origin = dict(path=candidate['source_selection'], hashes=candidate['local_hashes']) if candidate.get('source_selection') else None
        save(folder / PROFILE_FILE, dict(schema_version=2, model=model, assets=assets,
                                        contract=_comfy_contract(machine), stamps=stamps, origin=origin), durable=True)
        if direct:
            self._update('ready', candidate=model, done_bytes=candidate['bytes'] + candidate['copy_bytes'])
        return model

    def _smoke(self, machine, folder, cache, descriptor):
        from .managed import command_for
        attempt = folder / ('smoke-' + str(time.time_ns()))
        attempt.mkdir()
        prompt, output = attempt / 'prompt.txt', attempt / 'video.mp4'
        prompt.write_text('A blue ceramic cup is still on a table in soft daylight. Quiet room tone.\n',
                          encoding='utf-8')
        command = command_for(machine, 'generate', ['--cache', str(cache), '--prompt-file', str(prompt),
                              '--out', str(output), '--width', '256', '--height', '256', '--frames', '39',
                              '--no-two-pass', '--no-tuning'])
        self._child(machine, folder, 'smoke', command[1:], descriptor=descriptor, timeout=SMOKE_TIMEOUT)
        if (not output.is_file() or output.stat().st_size < 32
                or _read(output.with_suffix('.request.json')).get('success') is not True):
            raise ValueError('The candidate model did not produce a playable video.')
        self._child(machine, folder, 'smoke-validation', ['-m', WORKER_MODULE, 'validate-video', str(output)],
                    timeout=30)

    def _already_active(self, identity):
        return (self._machine().get(ACTIVE_KEY) or {}).get('id') == identity

    def activate(self, identity):
        if identity != OFFICIAL_ID:
            profile_folder(self.root, identity)
        return self._switch_when_idle(identity)

    def _activate(self, identity):
        machine = self._machine()
        if identity == (machine.get(ACTIVE_KEY) or {}).get('id', OFFICIAL_ID):
            return
        if identity == OFFICIAL_ID:
            original = machine[OFFICIAL_KEY]
            _read(Path(original['cache']) / 'manifest.json')
            machine.update(original)
            machine.pop(ACTIVE_KEY, None)
            machine.pop(OFFICIAL_KEY, None)
        else:
            receipt, cache = _receipt(profile_folder(self.root, identity))
            if cache is None:
                _verified_comfy_profile(self.root, receipt, machine)
            else:
                if receipt.get('contract') != _contract(machine):
                    raise ValueError('This model was tested on a different FreeVideo configuration. Import it again.')
                manifest = _read(cache / 'manifest.json')
                if _stamp_cache(cache, manifest) != receipt['stamps']:
                    raise ValueError('The installed video model changed. Inspect and import it again.')
            if not machine.get(ACTIVE_KEY):
                machine[OFFICIAL_KEY] = _baseline(machine)
            if cache is None:
                machine.update(machine[OFFICIAL_KEY])
            else:
                machine.update(cache=str(cache), prepared_format='int8_convrot' if manifest['precision'] == 'int8' else 'fp8')
            machine[ACTIVE_KEY] = receipt['model']
        save(self.root / 'machine.json', machine, durable=True)


def _transfer(request):
    from . import model_transfer, network, provision
    folder = request.parent
    spec = _read(request)
    status = asset_path(folder, spec['status'])
    ledger_path = folder / 'verified-downloads.json'
    ledger = load_ledger(ledger_path)
    done = 0
    try:
        # New repositories always use the official HF endpoint. Retain verified files and resumable partials.
        plan = dict(root=str(folder), network=dict(mode='official', proxy_mode=spec.get('proxy_mode', 'auto')))
        for row in spec['files']:
            path = asset_path(folder, row['target'])
            if not provision.verified(path, row, ledger):
                last = 0.

                def progress(amount, total, rate, **unused):
                    nonlocal last
                    now = time.monotonic()
                    if now - last >= 1. or amount == total:
                        save(status, dict(done=done + min(amount, row['bytes'])))
                        last = now

                # Keep a replaced or corrupt original available if provided before this import.
                if path.exists():
                    network.retain_partial(path, 'hash-rejected')
                model_transfer.download(row, path, plan, progress, provision.model_headers)
                ledger[str(path)] = provision.file_identity(path, row)
            save(ledger_path, ledger)
            done += row['bytes']
            save(status, dict(done=done))
    except Exception as error:
        token = os.environ.get('HF_TOKEN')
        message = str(error)
        if token:
            message = message.replace(token, '[redacted]')
        save(status, dict(done=done, error=(message or type(error).__name__)[:700]))
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    transfer = commands.add_parser('transfer')
    transfer.add_argument('request', type=Path)
    prepare = commands.add_parser('prepare')
    for name in ('base', 'checkpoint', 'cache_root', 'out'):
        prepare.add_argument(name, type=Path)
    validate = commands.add_parser('validate-video')
    validate.add_argument('video', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'transfer':
        _transfer(args.request)
    elif args.command == 'validate-video':
        import av
        for media in ('video', 'audio'):
            with av.open(str(args.video)) as container:
                if not getattr(container.streams, media) or next(container.decode(**{media: 0}), None) is None:
                    raise ValueError('The candidate video has no decodable %s stream.' % media)
    else:
        import torch
        from .fp8 import prepare_streamed
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        cache = prepare_streamed(args.base, args.checkpoint, args.cache_root)
        save(args.out, dict(cache=cache.relative_to(args.cache_root).as_posix()), durable=True)


if __name__ == '__main__':
    main()
