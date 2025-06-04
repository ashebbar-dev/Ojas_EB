import re, textwrap, json, urllib.request
from typing import List
from IPython.display import Javascript, display

# ╭────────── USER CONFIG ──────────╮
LAB_TO_EXTRACT = int(input(":"))
SOURCE_URL     = "https://tinyurl.com/hs2h9v6z"   # raw GitHub or TinyURL
VERBOSE        = True
# ╰──────────────────────────────────╯


def log(msg, lvl="INFO"):
    if VERBOSE:
        print(f"[{lvl}] {msg}")


# ------------------------------------------------------------------
# 1⃣  Download the raw python file
# ------------------------------------------------------------------
def fetch_raw(url: str) -> str:
    log(f"Downloading: {url}")
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ------------------------------------------------------------------
# 2⃣  Split into programs that start with '#LabX'
# ------------------------------------------------------------------
def split_by_lab(raw: str, lab_no: int) -> List[str]:
    hdr_any  = re.compile(r"^#\s*Lab\d+\s*$", re.M)
    hdr_this = re.compile(rf"^#\s*Lab{lab_no}\s*$", re.M)

    hdr_pos  = [m.start() for m in hdr_any.finditer(raw)] + [len(raw)]
    hdr_pos.sort()

    chunks = []
    for i, pos in enumerate(hdr_pos[:-1]):          # last item is sentinel (EOF)
        header_line = raw[pos : raw.find("\n", pos) + 1]
        if not hdr_this.match(header_line):
            continue                                # other lab → skip
        chunk = raw[pos : hdr_pos[i + 1]]
        chunks.append(textwrap.dedent(chunk).rstrip())

    return chunks


# ------------------------------------------------------------------
# 3⃣  Insert fresh notebook cells via JavaScript
# ------------------------------------------------------------------
import nbformat, pathlib, datetime as _dt

def insert_cells(code_blocks: List[str]):
    if not code_blocks:
        log("Nothing to write.", "WARN")
        return

    nb = nbformat.v4.new_notebook()
    for code in code_blocks:
        nb.cells.append(nbformat.v4.new_code_cell(code))

    ts     = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    target = pathlib.Path(f'Lab{LAB_TO_EXTRACT}.ipynb').resolve()
    nbformat.write(nb, target.open('w', encoding='utf-8'))

    log(f"Wrote {len(code_blocks)} cell(s) to: {target}")
    print("\n⇨ Open that notebook manually to continue working.\n")
# ------------------------------------------------------------------
# 4⃣  Main driver
# ------------------------------------------------------------------
def main():
    try:
        raw = fetch_raw(SOURCE_URL)
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return

    programs = split_by_lab(raw, LAB_TO_EXTRACT)
    if not programs:
        log(f"No '#Lab{LAB_TO_EXTRACT}' programs found.", "ERROR")
        return

    log(f"Found {len(programs)} program(s) for Lab {LAB_TO_EXTRACT}. Adding cells …")
    insert_cells(programs)
    log("✔ Finished.")

main()