#!/usr/bin/env python3
"""Create a disposable, diagnostic-only iteration-timer source copy.

Usage: python3 instrument_iteration.py --source /absolute/frozen-source \
           --output /absolute/new-diagnostic-directory

The output contains source/, instrumentation.patch and receipt.json. Consume
source/ with existing build/run tools. No build or numerical run is performed.
Use --ffi-timers for Julia/FFI consumers to print the timer tree after solve.
Printing is outside native solver timing but inside frontend wall time.
Nested timers overlap their parents; these timings must not earn speed credit.
Source trees must contain regular files/directories only (no symlinks). All files,
including dependencies and artifacts if present, are copied and fingerprinted.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

TARGET = 'crates/solver/src/solver/core/solver.rs'
MACRO = 'crates/solver/src/timers/timers.rs'
FFI_TARGET = 'crates/ffi/src/lib.rs'
FFI_ANCHOR = '''    fn solve(&mut self) -> Result<()> {
        self.solved = false;
        self.solver.solve();
        self.solved = true;
        Ok(())
    }'''
FFI_PATCHED = FFI_ANCHOR.replace('        self.solved = true;', '''        // Diagnostic-only output, after native solve timing has been captured.
        if let Some(timers) = &self.solver.timers {
            timers.print();
        }
        self.solved = true;''')
EXPECTED_MACRO = '''macro_rules! timeit {
    ($timer:ident => $key:literal; $($tt:tt)+) => {

        $timer.start_as_current($key);
        $(
            $tt
        )+
        $timer.stop_current();
    }
}'''
PHASES = [
    ('diagnostic residual and info', '''            self.residuals.update(&self.variables, &self.data);

            //calculate duality gap (scaled)
            //--------------
            μ = self.variables.calc_mu(&self.residuals, &self.cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            self.info.save_scalars(μ, α, σ, iter);

            // convergence check and printing
            // --------------
            self.info.update(
                &mut self.data,
                &self.variables,
                &self.residuals,
                &timers);'''),
    ('diagnostic affine RHS', '''            self.step_rhs
                .affine_step_rhs(&self.residuals, &self.variables, &self.cones);'''),
    ('diagnostic combined RHS', '''                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    &mut self.step_lhs,
                    σ,
                    μ,
                    m
                );'''),
    ('diagnostic affine step length', '''                α = self.get_step_length(StepDirection::Affine, scaling);'''),
    ('diagnostic combined step length', '''            α = self.get_step_length(StepDirection::Combined,scaling);'''),
    ('diagnostic iterate save and add', '''            self.info.save_prev_iterate(&self.variables,&mut self.prev_vars);

            self.variables.add_step(&self.step_lhs, α);'''),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlinks are unsupported: {path}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = digest(path.read_bytes())
        elif not path.is_dir():
            raise ValueError(f'non-regular source entry: {path}')
    # Explicit algorithm: SHA256 of compact, sorted UTF-8 JSON path->SHA256 map.
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'sha256': digest(encoded), 'files': files}


def instrument(original, macro):
    if macro.count(EXPECTED_MACRO) != 1:
        raise ValueError('timeit macro differs from reviewed start/body/stop expansion')
    result = original
    for label, body in PHASES:
        if label == 'diagnostic residual and info' and 'self.residuals.update_with_pool(' in original:
            body = body.replace(
                'self.residuals.update(&self.variables, &self.data);',
                'self.residuals.update_with_pool(&self.variables, &self.data, self.cones.thread_pool());')
        if label in original or result.count(body) != 1:
            raise ValueError(f'expected exactly one uninstrumented match for {label}')
        # No introduced closure, return binding, or control-flow jump. The
        # existing macro borrows timers mutably only during start/stop calls;
        # info.update may borrow &timers in between. Assignments target the
        # same outer μ/α variables, so macro body scope cannot discard results.
        indent = body[:len(body) - len(body.lstrip(' '))]
        wrapped = (indent + f'timeit!{{timers => "{label}"; {{\n'
                   + body + '\n' + indent + '}}')
        result = result.replace(body, wrapped, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffi-timers', action='store_true',
                        help='print native timers after each FFI solve (adds frontend wall time)')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error('output must not already exist')
    output = output.resolve()
    if not source.is_dir() or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('source/output must be disjoint directories')
    before = identity(source)
    original = (source / TARGET).read_text()
    patched = instrument(original, (source / MACRO).read_text())
    changes = {TARGET: (original, patched)}
    if args.ffi_timers:
        ffi = (source / FFI_TARGET).read_text()
        if ffi.count(FFI_ANCHOR) != 1 or FFI_PATCHED in ffi:
            raise ValueError('expected exactly one uninstrumented FFI solve anchor')
        changes[FFI_TARGET] = (ffi, ffi.replace(FFI_ANCHOR, FFI_PATCHED, 1))
    patch = ''.join(''.join(difflib.unified_diff(old.splitlines(keepends=True),
        new.splitlines(keepends=True), fromfile='a/' + path, tofile='b/' + path))
        for path, (old, new) in changes.items())
    output.mkdir(parents=True, exist_ok=False)
    copied = output / 'source'
    shutil.copytree(source, copied, symlinks=True)
    if identity(copied) != before or identity(source) != before:
        raise RuntimeError('source changed during copy; discard diagnostic output')
    expected = dict(before['files'])
    for path, (_, new) in changes.items():
        (copied / path).write_text(new)
        expected[path] = digest(new.encode())
    after = identity(copied)
    if after['files'] != expected or identity(source) != before:
        raise RuntimeError('unexpected source mutation; discard diagnostic output')
    (output / 'instrumentation.patch').write_text(patch)
    receipt = {
        'purpose': 'diagnostic phase attribution only; no speed credit',
        'source': str(source), 'output_source': str(copied),
        'identity_algorithm': 'sha256(compact sorted UTF-8 JSON relative-path->file-sha256 map)',
        'before': before, 'after': after,
        'patch_sha256': digest(patch.encode()),
        'generator_sha256': digest(Path(__file__).read_bytes()),
        'timers': [label for label, _ in PHASES],
        'changed_files': sorted(changes),
        'ffi_timers': args.ffi_timers,
        'scope': 'Timeit wrappers and optional post-solve FFI timer printing; no settings or math changes',
        'limits': 'Nested times overlap parents; timer overhead included; optional FFI printing adds frontend wall time after native timing; no build/run qualification',
    }
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'receipt': str(output / 'receipt.json'),
                      'source_sha256': before['sha256'],
                      'instrumented_sha256': after['sha256']}, sort_keys=True))


if __name__ == '__main__':
    main()
