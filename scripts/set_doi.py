"""Write the Zenodo DOI of the archived release into the article, the citation file and the README.

    python scripts/set_doi.py 10.5281/zenodo.1234567

Reserve the DOI first (Zenodo: New upload, "Reserve DOI"), run this, rebuild the PDFs
(sh latex/methodsx/build.sh), build the archive (python scripts/make_release.py --zip),
commit and push, then upload release/repv-<version>.zip to the reserved Zenodo record.
Edits: the two DOI placeholders of latex/methodsx/main.tex, the doi field of CITATION.cff,
the DOI badge and the archive link of README.md, and the Archive URL of pyproject.toml.
Running it again with the same DOI changes nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def edit(rel: str, pairs: list[tuple[str, str]]) -> None:
    p = ROOT / rel
    t = p.read_text()
    n = t
    for old, new in pairs:
        n = n.replace(old, new)
    if n != t:
        p.write_text(n)
        print(f"updated {rel}")
    else:
        print(f"unchanged {rel}")


def main(argv: list[str]) -> None:
    if len(argv) != 1 or not re.fullmatch(r"10\.\d{4,9}/\S+", argv[0]):
        sys.exit("usage: python scripts/set_doi.py 10.5281/zenodo.NNNNNNN")
    doi = argv[0]
    url = f"https://doi.org/{doi}"
    edit("latex/methodsx/main.tex", [(r"\todo{Zenodo DOI}", rf"\href{{{url}}}{{doi:{doi}}}")])
    edit("CITATION.cff", [("# After archiving the GitHub release on Zenodo, add:\n# doi: 10.5281/zenodo.XXXXXXX\n",
                           f"doi: {doi}\n")])
    edit("README.md", [("<!-- DOI badge: added by scripts/set_doi.py -->",
                        f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)]({url})"),
                       ("they come from the Zenodo archive.", f"they come from the Zenodo archive ({url}).")])
    edit("pyproject.toml", [('Repository = "https://github.com/princepriyam/RePV"\n',
                             f'Repository = "https://github.com/princepriyam/RePV"\nArchive = "{url}"\n')]
         if "Archive =" not in (ROOT / "pyproject.toml").read_text() else [])
    left = (ROOT / "latex/methodsx/main.tex").read_text().count(r"\todo{Zenodo DOI}")
    print(f"DOI placeholders left in main.tex: {left}")


if __name__ == "__main__":
    main(sys.argv[1:])
