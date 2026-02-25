# pylandxml-converter

Convert LandXML TIN surfaces into 3D mesh files.

The converter reads point/face geometry from LandXML and exports each surface as a separate mesh file.

Supported output formats:
- `.obj`
- `.off`
- `.ply`
- `.stl`
- `.mesh`

## Install

Using `uv`:

```powershell
uv sync
```

## Usage

Show help:

```powershell
uv run python landxml_convert.py --help
```

Convert one file (default format: `obj`):

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml
```

Export multiple formats:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml -f obj -f ply -f stl
```

Export all formats:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml --all-formats
```

List surfaces without conversion:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml --list-surfaces
```

Export only selected surfaces:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml -s "PR_Design Surface"
```

Flip normals:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml --flip-normals
```

Apply origin/offset/scale transforms:

```powershell
uv run python landxml_convert.py 3-AWC66.00_PR.xml --origin-mode centroid --scale 0.001
uv run python landxml_convert.py 3-AWC66.00_PR.xml --origin-mode bottom-left --offset-x 10 --offset-y 20 --offset-z 5
```

## Output Naming

By default:
- Output directory is the input filename stem.
- Output filename prefix is the input filename stem.
- Each surface name is appended, sanitized for filesystem-safe naming.

Example:
- Input: `3-AWCC3.xml`
- Surface: `PR_Design Surface`
- Output file: `3-AWCC3\3-AWCC3_PR_Design_Surface.obj`

You can override:
- output directory with `--outdir/-o`
- prefix with `--prefix`
- add stable index numbering with `--index`

## CLI Options

- `INPUT_PATH`: LandXML file path.
- `--outdir, -o PATH`: Output directory (default: input filename stem).
- `--format, -f TEXT`: Repeat for multiple formats.
- `--all-formats`: Export all supported formats.
- `--surface, -s TEXT`: Export only named surfaces (repeatable).
- `--prefix TEXT`: Filename prefix (default: input filename stem).
- `--index / --no-index`: Prefix files with numeric order.
- `--list-surfaces`: Print surfaces and exit.
- `--flip-normals`: Reverse triangle winding.
- `--origin-mode TEXT`: `none`, `centroid`, `bottom-left`.
- `--offset-x FLOAT`: X translation after origin shift and scale.
- `--offset-y FLOAT`: Y translation after origin shift and scale.
- `--offset-z FLOAT`: Z translation after origin shift and scale.
- `--scale FLOAT`: Uniform scale factor.
