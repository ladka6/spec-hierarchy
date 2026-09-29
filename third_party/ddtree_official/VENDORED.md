Vendored copy of the official DDTree implementation
https://github.com/liranringel/ddtree, commit c96427a185677bf4133ed865dd1626a5041aef9b (2026-04-16), MIT license (see LICENSE).

Only change: the top-level imports in ddtree.py and dflash.py were made package-relative
(`from model import` -> `from .model import`, `from dflash import` -> `from .dflash import`),
because the name `dflash` collides with the z-lab `dflash` package this repo uses.
Everything else is unchanged. Needs `loguru` (pip install loguru).
