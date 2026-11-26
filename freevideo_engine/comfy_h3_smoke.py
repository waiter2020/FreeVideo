"""Run import verification in the local Windows ComfyUI queue."""
import asyncio
from concurrent.futures import TimeoutError as Pending
import os
from pathlib import Path
import shutil
import time
import uuid


def _cancel(queue, identity):
    import nodes
    with queue.mutex:
        if queue.delete_queue_item(lambda row: row[1] == identity):
            return
        if any(row[1] == identity for row in queue.get_current_queue()[0]):
            nodes.interrupt_processing()


def _wait_for_output(port, queue, identity, check_cancelled):
    deadline = time.monotonic() + 20 * 60
    stop = None
    while True:
        if stop is None:
            try:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError('ComfyUI H3 import test timed out.')
            except (InterruptedError, TimeoutError) as error:
                stop = error
                _cancel(queue, identity)
        history = port.history(identity)
        if history is not None:
            if stop is not None:
                raise stop
            if (history.get('status') or {}).get('status_str') != 'success':
                reason = port.generation_error(history, None)['message']
                raise ValueError('ComfyUI H3 import test failed: ' + reason)
            return history['outputs']
        if port.prompt_state(identity) == 'missing' and port.history(identity) is None:
            if stop is not None:
                raise stop
            raise RuntimeError('ComfyUI no longer has the H3 import test in its queue.')
        time.sleep(.25)


def _run(server, profile, output, check_cancelled):
    from .agent.comfy import ComfyPort
    from .comfy_h3_runtime import SMOKE_OUTPUT_NODE, allow_smoke, smoke_graph

    check_cancelled()
    if not _available(server):
        raise ValueError('ComfyUI H3 import needs a compatible local Windows ComfyUI generation queue.')
    loop = server.loop
    if not loop.is_running():
        raise ValueError('The local ComfyUI queue is unavailable.')
    port = ComfyPort(server)
    run = 'FreeVideo/.model-smoke/' + uuid.uuid4().hex
    folder = port.output_path(run)
    profile = dict(profile, id=uuid.uuid4().hex)
    try:
        with allow_smoke(profile):
            graph = smoke_graph(profile['id'], variant=profile['preset'], frames=39, filename_prefix=run + '/video')
            ticket = asyncio.run_coroutine_threadsafe(port.submit(graph, {}), loop)
            while True:
                try:
                    identity = ticket.result(timeout=.25)
                    break
                except Pending:
                    if ticket.done() or not loop.is_running():
                        raise
            outputs = _wait_for_output(port, server.prompt_queue, identity, check_cancelled)
            check_cancelled()
            previews = outputs.get(SMOKE_OUTPUT_NODE, {}).get('images', [])
            if len(previews) != 1 or previews[0].get('subfolder', '').replace('\\', '/') != run:
                raise RuntimeError('The ComfyUI H3 import test did not return its video.')
            video = port.output_path(run + '/' + previews[0]['filename'])
            if video.parent != folder or video.suffix != '.mp4' or video.stat().st_size < 32:
                raise RuntimeError('The ComfyUI H3 import test did not produce a playable video.')
            shutil.copyfile(video, output)
            return Path(output)
    finally:
        if folder.is_dir():
            for item in folder.glob('video_*.mp4'):
                item.unlink(missing_ok=True)
            try:
                folder.rmdir()
            except OSError:
                pass


def _available(server):
    return (os.name == 'nt' and getattr(server, '_freevideo_front', None) is None
            and getattr(server, '_freevideo_h3_lifecycle', None) is not None)


def smoke_runner(server):
    if os.name != 'nt' or getattr(server, '_freevideo_front', None) is not None:
        return None

    def run(profile, output, check_cancelled):
        return _run(server, profile, output, check_cancelled)

    run.available = lambda: _available(server)
    return run
