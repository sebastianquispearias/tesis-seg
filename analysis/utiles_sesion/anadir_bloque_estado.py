"""Append the session block to ESTADO_SESION.md matching its own line endings.

The destination is inspected first, because appending Windows endings to a file
that uses Unix ones leaves it mixed, which is what happened earlier today with
the question bank. A backup is kept and the append is checked for reversibility.
"""
import shutil

DESTINO = r"G:\My Drive\UNM_vertebras_seg_v3\ESTADO_SESION.md"
FRAGMENTO = (r"C:\Users\User\AppData\Local\Temp\claude\G--"
             r"\9babba08-8fd8-4be2-afbe-c67e87bbda03\scratchpad\bloque_estado.md")
COPIA = DESTINO + ".ANTES_bloque_pool"

MARCA = "4/9 TARDE - EL POOL NO SE USO"


def finales(datos):
    cr = crlf = lf = 0
    for i, b in enumerate(datos):
        if b == 0x0D:
            if i + 1 < len(datos) and datos[i + 1] == 0x0A:
                crlf += 1
            else:
                cr += 1
        if b == 0x0A and (i == 0 or datos[i - 1] != 0x0D):
            lf += 1
    return cr, lf, crlf


with open(DESTINO, "rb") as fh:
    antes = fh.read()

if MARCA in antes.decode("utf-8"):
    raise SystemExit("el bloque ya esta: no se toca nada")

cr0, lf0, crlf0 = finales(antes)
usa_crlf = crlf0 > lf0
print("DESTINO: CR sueltos {}  LF sueltos {}  CRLF {}  ->  usa {}".format(
    cr0, lf0, crlf0, "CRLF" if usa_crlf else "LF"))

shutil.copy2(DESTINO, COPIA)

with open(FRAGMENTO, "r", encoding="utf-8", newline="") as fh:
    frag = fh.read().replace("\r\n", "\n")
if usa_crlf:
    frag = frag.replace("\n", "\r\n")

despues = antes + frag.encode("utf-8")
with open(DESTINO, "wb") as fh:
    fh.write(despues)

with open(DESTINO, "rb") as fh:
    escrito = fh.read()
cr, lf, crlf = finales(escrito)

print()
print("copia            :", COPIA)
print("lineas antes     :", antes.count(b"\n"))
print("lineas despues   :", escrito.count(b"\n"))
print("reversible       :", "SI" if escrito[:len(antes)] == antes else "NO")
print("CR sueltos       :", cr)
print("LF sueltos       :", lf)
print("CRLF             :", crlf)
print("mezcla de finales:", "SI, MAL" if (lf > 0 and crlf > 0) else "no")
print("bloque presente  :", MARCA in escrito.decode("utf-8"))
