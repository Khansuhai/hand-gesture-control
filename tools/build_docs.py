"""Build the PDFs in docs/ with Microsoft Edge (headless).

    docs/cheat-sheet.html            -> docs/gesture-cheat-sheet.pdf
    docs/user-guide-template.html    -> docs/user-guide.html -> docs/user-guide.pdf
      (the guide = 3 step-by-step pages + the cheat sheet pages)

Run: .venv\\Scripts\\python.exe tools\\build_docs.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def print_pdf(html: Path, pdf: Path) -> None:
    edge = next((p for p in EDGE_PATHS if p.exists()), None)
    if edge is None:
        raise SystemExit("Microsoft Edge not found - open the .html file and use Print > Save as PDF instead.")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [str(edge), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
             "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
            check=True, timeout=120,
        )
    data = pdf.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    print(f"{pdf.name}: {pages} pages, {len(data) // 1024} KB")


def main():
    sheet_html = (DOCS / "cheat-sheet.html").read_text(encoding="utf-8")
    style = re.search(r"<style>.*?</style>", sheet_html, re.S).group(0)
    body = re.search(r"<body>(.*)</body>", sheet_html, re.S).group(1)
    body = body.replace("Page 1 of 2", "Cheat sheet 1 of 2").replace("Page 2 of 2", "Cheat sheet 2 of 2")
    guide = (DOCS / "user-guide-template.html").read_text(encoding="utf-8")
    guide = guide.replace("<!--STYLE-->", style).replace("<!--CHEATSHEET-->", body)
    (DOCS / "user-guide.html").write_text(guide, encoding="utf-8")

    print_pdf(DOCS / "cheat-sheet.html", DOCS / "gesture-cheat-sheet.pdf")
    print_pdf(DOCS / "user-guide.html", DOCS / "user-guide.pdf")


if __name__ == "__main__":
    main()
