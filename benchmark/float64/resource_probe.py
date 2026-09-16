#!/usr/bin/env python3
"""Record wait4 high-water RSS for one owned child; preserve its output and exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def write_receipt(path, receipt):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        os.replace(temporary, path)
    except OSError:
        # Preserve the utility's stderr/status. The controller records a missing
        # memory receipt separately instead of inventing a peak measurement.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv):
    if len(argv) < 3 or argv[1] != '--':
        raise SystemExit('usage: resource_probe.py RECEIPT -- COMMAND [ARG ...]')
    output, command = Path(argv[0]), argv[2:]
    if not hasattr(os, 'wait4') or sys.platform not in ('darwin', 'linux'):
        raise SystemExit('native RSS receipt requires wait4 on macOS or Linux')
    try:
        # Inherit the controller's stdout/stderr and process group. The retained
        # watchdog therefore owns both this probe and the numerical child.
        process = subprocess.Popen(command)
    except OSError as error:
        write_receipt(output, {'complete': False, 'command': command,
                              'launch_error': f'{type(error).__name__}: {error}'})
        return 127
    while True:
        try:
            _, status, usage = os.wait4(process.pid, 0)
            break
        except InterruptedError:
            continue
    if hasattr(os, 'waitstatus_to_exitcode'):
        code = os.waitstatus_to_exitcode(status)
    elif os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        code = -os.WTERMSIG(status)
    else:
        code = 1
    # wait4 already reaped this exact child. Avoid Popen trying to wait again.
    process.returncode = code
    unit = 1 if sys.platform == 'darwin' else 1024
    peak_bytes = int(usage.ru_maxrss) * unit
    write_receipt(output, {
        'complete': True, 'command': command, 'child_pid': process.pid,
        'child_exit_code': code, 'platform': sys.platform,
        'source': 'wait4.ru_maxrss', 'raw_ru_maxrss': usage.ru_maxrss,
        'raw_unit': 'bytes' if unit == 1 else 'KiB',
        'max_rss_bytes': peak_bytes, 'max_rss_mib': peak_bytes / 1048576,
        'user_cpu_s': usage.ru_utime, 'system_cpu_s': usage.ru_stime,
        'scope': 'single solver child process high-water RSS; not simultaneous process-group peak',
    })
    if code < 0:
        sig = -code
        try:
            signal.signal(sig, signal.SIG_DFL)
        except (OSError, ValueError):
            pass  # SIGKILL cannot have a handler.
        os.kill(os.getpid(), sig)
        os._exit(128 + sig)  # Fallback only if the signal did not terminate us.
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
