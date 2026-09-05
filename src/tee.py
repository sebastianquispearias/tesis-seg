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
gets abandoned. This module duplicates: every line goes to the file and to the
original stream, so the notebook keeps displaying its output exactly as before.

Writing straight to a mounted Google Drive does not work for this. Drive
publishes a file when it is closed, so a log held open for the two hours of a
run only appears once the run ends, and is lost if the session is killed first,
which is precisely the case worth covering. The transcript is therefore written
to local disk, where a write is immediate, and copied over to its destination
every few seconds and once more at the end. A session that dies leaves behind
everything up to the last copy.

Typical use, around the training call::

    from src.tee import tee_output

    with tee_output(os.path.join(cfg["exp_dir"], "stdout.log")):
        artifacts = run_training(cfg, loaders)
        results = evaluate_checkpoint(...)
"""

import contextlib
import os
import re
import shutil
import sys
import tempfile
import time

INTERVALO_COPIA = 20.0      # segundos entre copias al destino


class _Tee:
    """A writable stream that forwards to several streams at once."""

    def __init__(self, primary, *others, al_escribir=None):
        self._primary = primary
        self._streams = (primary,) + others
        self._al_escribir = al_escribir

    def write(self, data):
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                # a broken secondary must never take down the run
                pass
        if self._al_escribir is not None:
            try:
                self._al_escribir()
            except Exception:
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


def _nombre_local(destino):
    """A local scratch name derived from the destination, safe on any platform."""
    partes = os.path.abspath(destino).replace("\\", "/").split("/")
    etiqueta = "_".join(partes[-3:]) if len(partes) >= 3 else partes[-1]
    etiqueta = re.sub(r"[^A-Za-z0-9_.-]", "_", etiqueta)
    return os.path.join(tempfile.gettempdir(), "tee_" + etiqueta)


@contextlib.contextmanager
def tee_output(destino, also_stderr=True, cabecera=True,
               intervalo=INTERVALO_COPIA):
    """Duplicate stdout, and optionally stderr, into destino while still printing.

    The destination is opened in append mode, so a run repeated in the same
    directory adds to its history instead of erasing it.
    """
    destino = os.path.abspath(destino)
    os.makedirs(os.path.dirname(destino), exist_ok=True)

    local = _nombre_local(destino)
    if os.path.isfile(destino):
        try:
            shutil.copyfile(destino, local)      # conservar lo ya escrito
        except Exception:
            pass
    elif os.path.isfile(local):
        try:
            os.remove(local)                     # sobra de un run anterior
        except Exception:
            pass

    fh = open(local, "a", encoding="utf-8", buffering=1)

    if cabecera:
        fh.write("\n{}\n{}  {}\n{}\n".format(
            "=" * 78,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            os.path.basename(os.path.dirname(destino)),
            "=" * 78))
        fh.flush()

    estado = {"ultima": 0.0}

    def copiar():
        try:
            fh.flush()
            shutil.copyfile(local, destino)
            estado["ultima"] = time.monotonic()
        except Exception:
            pass

    def quiza_copiar():
        if time.monotonic() - estado["ultima"] >= intervalo:
            copiar()

    copiar()                                     # que exista desde el principio

    out_antes, err_antes = sys.stdout, sys.stderr
    sys.stdout = _Tee(out_antes, fh, al_escribir=quiza_copiar)
    if also_stderr:
        sys.stderr = _Tee(err_antes, fh, al_escribir=quiza_copiar)
    try:
        yield destino
    except BaseException:
        # the traceback is the most valuable thing a dead run leaves behind
        import traceback
        try:
            fh.write("\n--- EXCEPCION ---\n")
            traceback.print_exc(file=fh)
        except Exception:
            pass
        raise
    finally:
        sys.stdout, sys.stderr = out_antes, err_antes
        try:
            copiar()
        finally:
            try:
                fh.close()
            except Exception:
                pass
