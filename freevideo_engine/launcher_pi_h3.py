"""Pi-H3 downloads for the launcher's selected installation, with no ComfyUI connection."""
import json
import os
from pathlib import Path
from threading import Event, Thread
import time

from . import fast_pack
from .fast_tier import MODES, mode_spec
from .long_paths import io_path


_REASONS = {
    'gpu': ('Requires an NVIDIA CUDA GPU', '需要 NVIDIA CUDA 显卡'),
    'engine': ('Requires the FreeVideo INT8 video model', '需要 FreeVideo INT8 视频模型'),
    'mac': ('Unavailable on Mac', 'Mac 暂不支持'),
    'selected_component_changed': ('Repair the selected files in Model management', '请在模型管理中修复所选文件'),
    'busy': ('Another download is in progress', '有其他模型正在下载'),
    'offline': ('Network unavailable', '网络不可用'),
    'disk_space': ('Not enough disk space', '磁盘空间不足'),
    'integrity_failed': ('File verification failed', '文件校验失败'),
    'write_failed': ('Could not save the model files', '无法保存模型文件'),
    'not_published': ('Download unavailable', '暂无可用下载'),
    'interrupted': ('Download interrupted', '下载已中断'),
}


class PiH3Downloads:
    def __init__(self):
        self.engine = self.location = self.installer = None
        self.machine = None
        self.thread = self.result = self.pending = None
        self.values, self.errors = {}, {}
        self.next_poll = 0.
        self.starting = False
        self.paused = Event()

    @property
    def busy(self):
        return bool(self.starting or (self.pending and self.pending[0] == 'install')
                    or (self.installer and self.installer.owns_download()))

    def request(self, action, mode, job_id):
        if self.engine is None or self.pending is not None or action not in ('install', 'cancel') or mode not in MODES:
            return
        if action == 'install':
            self.paused.clear()
        self.errors.pop(mode, None)
        self.pending = (action, mode, job_id)

    def pause_owned(self):
        self.paused.set()
        if self.pending and self.pending[0] == 'install':
            self.pending = None
        self.next_poll = 0.

    def close(self, timeout=5):
        deadline = time.monotonic() + timeout
        self.pause_owned()
        if self.thread is not None:
            self.thread.join(max(0., deadline - time.monotonic()))
        if self.installer is not None:
            self.installer.close(timeout=max(0., deadline - time.monotonic()))

    def poll(self, engine):
        engine = Path(engine) if engine else None
        if engine != self.engine:
            self.pause_owned()
            self.engine, self.values, self.errors, self.pending = engine, {}, {}, None
            self.next_poll = 0.
        if self.thread is not None:
            if self.thread.is_alive():
                return
            self.thread = None
            self.starting = False
            if self.result is not None:
                selected, values, failed = self.result
                if selected == engine:
                    self.values = values
                    if failed is not None:
                        mode, reason = failed
                        self.errors[mode] = reason
        now = time.monotonic()
        if (engine is None and not self.busy) or (self.pending is None and now < self.next_poll):
            return
        action, self.pending = self.pending, None
        self.result = None
        self.starting = action is not None and action[0] == 'install'
        active = self.busy or any(row.get('install', {}).get('state') == 'running' for row in self.values.values())
        self.next_poll = now + (.5 if active else 2.)
        self.thread = Thread(target=self._poll, args=(engine, action), name='freevideo-launcher-pi-h3', daemon=True)
        self.thread.start()

    def _poll(self, engine, command):
        failed = None
        try:
            if self.paused.is_set() and self.installer is not None:
                self.installer.cancel_owned()
            if engine is None:
                self.result = (None, {}, None)
                return
            machine = json.loads(io_path(engine / 'machine.json').read_text(encoding='utf-8'))
            if not isinstance(machine, dict) or machine.get('ready') is not True:
                raise ValueError('Installation is not ready')
            root = Path(os.environ.get('FREEVIDEO_MODEL_ROOT') or machine.get('model_root') or engine / 'models').expanduser()
            location = (engine, root)
            self.machine = machine
            if self.location != location:
                if self.installer is not None:
                    self.installer.close(timeout=0)
                self.installer = fast_pack.Installer(lambda: root, engine_root=engine, machine=lambda: self.machine)
                self.location = location
            installer = self.installer
            if command is not None:
                action, mode, job_id = command
                try:
                    if action == 'cancel':
                        installer.cancel(mode, job_id=job_id)
                    elif not self.paused.is_set() and self.engine == engine:
                        installer.start(mode)
                        if self.paused.is_set() or self.engine != engine:
                            installer.cancel_owned()
                except fast_pack.PackError as error:
                    failed = (mode, error.reason)
            values = {mode: installer.status(mode) for mode in MODES}
        except (OSError, TypeError, ValueError, fast_pack.PackError):
            values = {mode: dict(compatible=False, compatibility_reason='engine') for mode in MODES}
        self.result = (engine, values, failed)

    def view(self, t, *, can_start):
        rows = []
        for mode in MODES:
            value = self.values.get(mode, {})
            install = value.get('install', {})
            running = install.get('state') == 'running'
            verify = install.get('phase') == 'verify'
            cancelling = install.get('cancelling') is True
            present = value.get('present') is True
            compatible = value.get('compatible') is not False
            resume = value.get('resume_bytes') or 0
            remaining = max(0, (value.get('bytes') or 0) - resume)
            blocked = value.get('blocked_by')
            reason = install.get('reason') if install.get('state') == 'failed' else self.errors.get(mode)
            cancelled = install.get('state') == 'cancelled'
            resumable = cancelled or reason == 'interrupted' or resume > 0 and not reason
            if running:
                detail = (('Cancelling…', '正在取消…') if cancelling else ('Verifying files', '正在校验文件')
                          if verify else ('Downloading', '正在下载'))
            elif not compatible:
                detail = _REASONS.get(value.get('compatibility_reason'), _REASONS['engine'])
            elif present:
                detail = ('Installed', '已安装')
            elif not value:
                detail = ('Checking files', '正在检查文件')
            elif not value.get('published'):
                detail = _REASONS['not_published']
            elif blocked:
                detail = _REASONS['busy']
            elif reason:
                detail = _REASONS.get(reason, ('Could not download the model', '无法下载模型'))
            elif resumable:
                detail = ('Paused', '已暂停')
            else:
                detail = ('Not installed', '未安装')
            total = install.get('verify_total' if verify else 'total') or 0
            done = install.get('verify_done' if verify else 'done') or 0
            steps = mode_spec(mode)['steps']
            action = ('Continue', '继续') if resumable else ('Retry', '重试') if reason else ('Download', '下载')
            rows.append(dict(mode=mode, title=t('pi-H3 · {steps} steps', 'pi-H3 · {steps} 步', {'steps': steps}),
                             detail=t(*detail), bytes=remaining, running=running, error=bool(reason and not present),
                             fraction=min(1., done / total) if total else -1., action=t(*action),
                             show_download=bool(value and compatible and value.get('published') and not present and not running),
                             can_start=bool(can_start and not blocked and not self.starting and self.pending is None),
                             can_cancel=bool(running and not cancelling and install.get('job_id') and self.pending is None),
                             job_id=install.get('job_id') or ''))
        return dict(visible=self.engine is not None, busy=self.busy, rows=rows)
