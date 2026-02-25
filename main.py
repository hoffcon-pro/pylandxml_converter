#!/usr/bin/env python3
"""
landxml2mesh.py

Convert LandXML TIN surfaces to 3D mesh files (one file per surface).

Reads:
  LandXML -> Surfaces/Surface/Definition/TIN/Pnts + Faces

Writes (via PyMesh):
  .obj, .off, .ply, .stl, .mesh  (format support depends on your PyMesh build)

Examples:
  python landxml2mesh.py input.xml -o out --formats obj ply stl
  python landxml2mesh.py input.xml -o out --surfaces "Existing Ground" "Design"
  python landxml2mesh.py input.xml -o out --all-formats
"""

from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError:
    print("ERROR: numpy is required. pip install numpy", file=sys.stderr)
    raise

try:
    import pymesh
except ImportError:
    pymesh = None

try:
    import typer
except ImportError:
    print("ERROR: typer is required. pip install typer", file=sys.stderr)
    raise


# -----------------------------
# Helpers
# -----------------------------

def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = name.strip()
    if not name:
        name = "surface"
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:max_len]


def landxml_ns(root: ET.Element) -> Dict[str, str]:
    """
    ElementTree needs namespaces for .find/.findall on namespaced XML.

    If the document has a default namespace like:
      <LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2">
    then tags are in that namespace.
    """
    m = re.match(r"\{(.+)\}", root.tag)
    if m:
        return {"lx": m.group(1)}
    return {}  # no namespace


def qname(ns: Dict[str, str], tag: str) -> str:
    """Qualify a tag name with expanded namespace form if present."""
    uri = ns.get("lx")
    return f"{{{uri}}}{tag}" if uri else tag


def text_to_floats(s: str) -> List[float]:
    return [float(x) for x in s.strip().split()]


def text_to_ints(s: str) -> List[int]:
    return [int(x) for x in s.strip().split()]


# -----------------------------
# LandXML parsing
# -----------------------------

def parse_surface_tin(
    surface_el: ET.Element, ns: Dict[str, str]
) -> Optional[Tuple[NDArray[np.float64], NDArray[np.int64]]]:
    """
    Extract vertices and faces from:
      Surface/Definition/TIN/Pnts/P
      Surface/Definition/TIN/Faces/F

    Vertices:
      Each <P id="123">x y z</P> (sometimes y x z depending on authoring; LandXML uses Easting Northing Elevation)
    Faces:
      Each <F>i j k</F> referencing P ids

    Returns:
      (V, F) where:
        V is (n,3) float64
        F is (m,3) int64, 0-based indexing
    """
    # Find TIN
    def_el = surface_el.find(f".//{qname(ns,'Definition')}")
    if def_el is None:
        return None

    tin_el = def_el.find(f".//{qname(ns,'TIN')}")
    # Many LandXML exports store Pnts/Faces directly under Definition with surfType="TIN".
    if tin_el is None:
        tin_el = def_el

    # Points
    pnts_el = tin_el.find(f".//{qname(ns,'Pnts')}")
    if pnts_el is None:
        return None

    pid_to_index: Dict[int, int] = {}
    verts: List[List[float]] = []

    for p_el in pnts_el.findall(qname(ns, "P")):
        pid_str = p_el.attrib.get("id") or p_el.attrib.get("name") or p_el.attrib.get("pntRef")
        if pid_str is None:
            # Some exporters omit ids; then faces typically won't work either.
            continue

        try:
            pid = int(pid_str)
        except ValueError:
            # If pid isn't int, skip (faces likely won't match)
            continue

        if p_el.text is None:
            continue

        coords = text_to_floats(p_el.text)
        if len(coords) < 3:
            continue
        xyz = coords[:3]
        pid_to_index[pid] = len(verts)
        verts.append(xyz)

    if not verts:
        return None

    V = np.asarray(verts, dtype=np.float64)

    # Faces
    faces_el = tin_el.find(f".//{qname(ns,'Faces')}")
    if faces_el is None:
        return None

    F_list: List[List[int]] = []
    missing_refs = 0

    for f_el in faces_el.findall(qname(ns, "F")):
        if f_el.text is None:
            continue
        tri = text_to_ints(f_el.text)
        if len(tri) < 3:
            continue
        a, b, c = tri[:3]

        try:
            ia = pid_to_index[a]
            ib = pid_to_index[b]
            ic = pid_to_index[c]
        except KeyError:
            missing_refs += 1
            continue

        F_list.append([ia, ib, ic])

    if not F_list:
        return None

    if missing_refs:
        print(f"  warning: skipped {missing_refs} faces due to missing point references", file=sys.stderr)

    F = np.asarray(F_list, dtype=np.int64)
    return V, F


def find_surfaces(root: ET.Element, ns: Dict[str, str]) -> List[ET.Element]:
    """
    Find all Surface elements in the LandXML.
    Typical path:
      LandXML/Surfaces/Surface
    But some exports nest Surfaces under other nodes; so search broadly.
    """
    # Prefer canonical location first
    canonical = root.findall(f".//{qname(ns,'Surfaces')}/{qname(ns,'Surface')}")
    if canonical:
        return canonical
    # Fallback: any Surface tag
    return root.findall(f".//{qname(ns,'Surface')}")


def surface_name(surface_el: ET.Element) -> str:
    # Common attributes: name, desc
    return surface_el.attrib.get("name") or surface_el.attrib.get("desc") or surface_el.attrib.get("id") or "surface"


# -----------------------------
# Export
# -----------------------------

SUPPORTED_FORMATS = {"obj", "off", "ply", "stl", "mesh"}
HAS_PYMESH_API = bool(
    pymesh
    and hasattr(pymesh, "form_mesh")
    and hasattr(pymesh, "save_mesh")
)


def _write_obj(V: NDArray[np.float64], F: NDArray[np.int64], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for x, y, z in V:
            f.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in F:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _write_off(V: NDArray[np.float64], F: NDArray[np.int64], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for x, y, z in V:
            f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in F:
            f.write(f"3 {a} {b} {c}\n")


def _write_ply(V: NDArray[np.float64], F: NDArray[np.int64], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in V:
            f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in F:
            f.write(f"3 {a} {b} {c}\n")


def _triangle_normal(
    v0: NDArray[np.float64], v1: NDArray[np.float64], v2: NDArray[np.float64]
) -> NDArray[np.float64]:
    n = np.cross(v1 - v0, v2 - v0)
    nrm = np.linalg.norm(n)
    if nrm == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return n / nrm


def _write_stl_ascii(
    V: NDArray[np.float64], F: NDArray[np.int64], out_path: str, solid_name: str
) -> None:
    safe_name = sanitize_filename(solid_name) or "surface"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"solid {safe_name}\n")
        for a, b, c in F:
            v0 = V[a]
            v1 = V[b]
            v2 = V[c]
            nx, ny, nz = _triangle_normal(v0, v1, v2)
            f.write(f"  facet normal {nx:.12g} {ny:.12g} {nz:.12g}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.12g} {v0[1]:.12g} {v0[2]:.12g}\n")
            f.write(f"      vertex {v1[0]:.12g} {v1[1]:.12g} {v1[2]:.12g}\n")
            f.write(f"      vertex {v2[0]:.12g} {v2[1]:.12g} {v2[2]:.12g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {safe_name}\n")


def _write_medit_mesh(V: NDArray[np.float64], F: NDArray[np.int64], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")
        f.write("Vertices\n")
        f.write(f"{len(V)}\n")
        for x, y, z in V:
            f.write(f"{x:.12g} {y:.12g} {z:.12g} 0\n")
        f.write("\nTriangles\n")
        f.write(f"{len(F)}\n")
        for a, b, c in F:
            f.write(f"{a + 1} {b + 1} {c + 1} 0\n")
        f.write("\nEnd\n")


def export_mesh(
    V: NDArray[np.float64], F: NDArray[np.int64], out_path: str, surface: str
) -> None:
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported output format: {ext}")

    if HAS_PYMESH_API:
        try:
            mesh = pymesh.form_mesh(V, F)
            pymesh.save_mesh(out_path, mesh, ascii=True)
            return
        except Exception:
            # Fallback writers keep conversion usable when the installed pymesh
            # package is not the full geometry IO build.
            pass

    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("Only triangular faces are supported (m,3).")
    if ext == "obj":
        _write_obj(V, F, out_path)
    elif ext == "off":
        _write_off(V, F, out_path)
    elif ext == "ply":
        _write_ply(V, F, out_path)
    elif ext == "stl":
        _write_stl_ascii(V, F, out_path, surface)
    elif ext == "mesh":
        _write_medit_mesh(V, F, out_path)


# -----------------------------
# CLI
# -----------------------------

def _normalize_formats(formats: Sequence[str], all_formats: bool) -> List[str]:
    if all_formats:
        fmts = sorted(SUPPORTED_FORMATS)
        return [f for f in fmts if f in SUPPORTED_FORMATS]

    fmts = [f.lower().lstrip(".") for f in formats]
    bad = [f for f in fmts if f not in SUPPORTED_FORMATS]
    if bad:
        allowed = ", ".join(sorted(SUPPORTED_FORMATS))
        raise typer.BadParameter(f"Unsupported format(s): {bad}. Choose from: {allowed}")
    return [f for f in fmts if f in SUPPORTED_FORMATS]


def run_conversion(
    input_path: Path,
    outdir: Optional[Path],
    formats: Sequence[str],
    all_formats: bool,
    selected_surfaces: Optional[Sequence[str]],
    prefix: Optional[str],
    index: bool,
    list_surfaces: bool,
) -> int:
    default_stem = sanitize_filename(input_path.stem)
    effective_outdir = outdir if outdir is not None else Path(default_stem)
    effective_prefix = prefix if prefix is not None else default_stem

    try:
        tree = ET.parse(input_path)
    except Exception as e:
        typer.echo(f"ERROR: Failed to parse XML: {e}", err=True)
        return 2

    root = tree.getroot()
    ns = landxml_ns(root)

    surface_elements = find_surfaces(root, ns)
    if not surface_elements:
        typer.echo("ERROR: No <Surface> elements found in the file.", err=True)
        return 1

    if list_surfaces:
        typer.echo(f"Found {len(surface_elements)} surface(s):")
        for idx, s_el in enumerate(surface_elements, start=1):
            typer.echo(f"{idx}. {surface_name(s_el)}")
        return 0

    if not HAS_PYMESH_API:
        typer.echo(
            "warning: full PyMesh API not detected; using built-in format writers.",
            err=True,
        )

    fmts = _normalize_formats(formats=formats, all_formats=all_formats)
    os.makedirs(effective_outdir, exist_ok=True)
    wanted = set(selected_surfaces) if selected_surfaces else None

    exported_any = False
    for idx, s_el in enumerate(surface_elements, start=1):
        name = surface_name(s_el)
        if wanted is not None and name not in wanted:
            continue

        typer.echo(f"Surface: {name}")

        parsed = parse_surface_tin(s_el, ns)
        if parsed is None:
            typer.echo("  skipped: no TIN/Pnts/Faces found or failed to parse", err=True)
            continue

        V, F = parsed
        base = sanitize_filename(name)
        if index:
            base = f"{idx:03d}_{base}"
        if effective_prefix:
            base = f"{sanitize_filename(effective_prefix)}_{base}"

        for ext in fmts:
            out_path = os.path.join(effective_outdir, f"{base}.{ext}")
            try:
                export_mesh(V, F, out_path, name)
                typer.echo(f"  wrote: {out_path}")
                exported_any = True
            except Exception as e:
                typer.echo(f"  failed: {out_path} ({e})", err=True)

    if wanted is not None and not exported_any:
        typer.echo(
            "WARNING: No surfaces were exported. Check that --surface names match exactly what is in the LandXML.",
            err=True,
        )

    return 0 if exported_any else 1


def main(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a LandXML file (.xml).",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        "-o",
        resolve_path=True,
        help="Output directory for generated meshes. Defaults to input filename stem.",
    ),
    formats: List[str] = typer.Option(
        ["obj"],
        "--format",
        "-f",
        help="Output format(s). Repeat this flag: -f obj -f ply -f stl.",
    ),
    all_formats: bool = typer.Option(
        False,
        "--all-formats",
        help="Export every supported format: obj, off, ply, stl, mesh.",
    ),
    surfaces: Optional[List[str]] = typer.Option(
        None,
        "--surface",
        "-s",
        help="Only export named surface(s). Repeat to include multiple surfaces.",
    ),
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        help="Filename prefix for exported surfaces. Defaults to input filename stem.",
    ),
    index: bool = typer.Option(
        False,
        "--index/--no-index",
        help="Prefix output names with a stable numeric index.",
    ),
    list_surfaces: bool = typer.Option(
        False,
        "--list-surfaces",
        help="List surfaces in the input file and exit without conversion.",
    ),
) -> None:
    """
    Convert LandXML surface geometry into 3D mesh files.

    The program reads TIN point/face data from one LandXML file and exports each
    surface as a separate mesh file in one or more formats (.obj, .off, .ply,
    .stl, .mesh). You can export all surfaces or only selected surface names.

    Examples:
      python main.py 3-AWC66.00_PR.xml -o out -f obj -f ply
      python main.py 3-AWC66.00_PR.xml --all-formats
      python main.py 3-AWC66.00_PR.xml -s "Existing Ground" -s "Design"
      python main.py 3-AWC66.00_PR.xml --list-surfaces
    """
    exit_code = run_conversion(
        input_path=input_path,
        outdir=outdir,
        formats=formats,
        all_formats=all_formats,
        selected_surfaces=surfaces,
        prefix=prefix,
        index=index,
        list_surfaces=list_surfaces,
    )
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    typer.run(main)
