"""Keep a copy of everything a run prints, without hiding it from the screen.

A run report records what a run measured; it does not record what the run said
while it was running. That transcript is where a training loop announces the
things no summary captures: that the unlabeled loader was missing, that a batch
came out the wrong shape, or the traceback of a run that died in its first
epoch. In Colab it lives only in the browser tab, and a notebook pushed to the
repository is stored without its outputs, so the transcript disappears with the
session.

The usual way of saving it, contextlib.redirect_stdout, replaces the stream
instead of duplicating it, which is why the screen goes blank and the practice
gets abandoned. This module duplicates: every line is written to the file and to
the original stream, so the notebook keeps displaying its output exactly as
before.

Every write is flushed, so a session that is killed by a timeout still leaves a
complete log up to the moment it stopped, which is precisely the case worth
having.

Typical use, around the training call::

    from src.tee import tee_output

    with tee_output(os.path.join(cfg["exp_dir"], "stdout.log")):
        artifacts = run_training(cfg, loaders)
        results = evaluate_checkpoint(...)
"""

import contextlib
import os
import sys
import time


class _Tee:
    """A writable stream that forwards to several streams at once."""

    def __init__(self, primary, *others):
        self._primary = primary
        self._streams = (primary,) + others

    def write(self, data):
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                # a broken secondary must never take down the run
                pass
        return n if n else len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self):
        # Some libraries ask for the underlying descriptor. Handing over the
        # original one means their output bypasses the copy, which is better
        # than raising in the middle of a run.
        return self._primary.fileno()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")


@contextlib.contextmanager
def tee_output(path, also_stderr=True, cabecera=True):
    """Duplicate stdout, and optionally stderr, into path while still printing.

    The file is opened in append mode so that a run resumed in the same
    directory adds to its history instead of erasing it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)

    if cabecera:
        fh.write("\n{}\n{}  {}\n{}\n".format(
            "=" * 78,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            os.path.basename(os.path.dirname(os.path.abspath(path))),
            "=" * 78))
        fh.flush()

    out_antes, err_antes = sys.stdout, sys.stderr
    sys.stdout = _Tee(out_antes, fh)
    if also_stderr:
        sys.stderr = _Tee(err_antes, fh)
    try:
        yield path
    except BaseException:
        # the traceback is the most valuable thing a dead run leaves behind
        import traceback
        try:
            fh.write("\n--- EXCEPCION ---\n")
            traceback.print_exc(file=fh)
            fh.flush()
        except Exception:
            pass
        raise
    finally:
        sys.stdout, sys.stderr = out_antes, err_antes
        try:
            fh.close()
        except Exception:
            pass
