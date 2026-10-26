"""Parsers for the MIRFLICKR metadata files (tags, EXIF, license) and annotations."""
from pathlib import Path


def _read_lines(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\r\n") for line in fh]
    except FileNotFoundError:
        return []


def read_tags(meta_dir, image_id):
    """Cleaned Flickr tags (lowercase, no spaces), one per line in tags/tagsN.txt."""
    lines = _read_lines(Path(meta_dir) / "tags" / f"tags{image_id}.txt")
    seen, tags = set(), []
    for line in lines:
        tag = line.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def read_exif(meta_dir, image_id):
    """EXIF fields: lines alternate '-Field name' and its value."""
    lines = _read_lines(Path(meta_dir) / "exif" / f"exif{image_id}.txt")
    exif = {}
    for i in range(0, len(lines) - 1, 2):
        key = lines[i].lstrip("-").strip()
        if key and key not in exif:
            exif[key] = lines[i + 1].strip()
    return exif


def read_license(meta_dir, image_id):
    """license/licenseN.txt holds 'Key : value' lines (title, owner, licence...)."""
    info = {}
    for line in _read_lines(Path(meta_dir) / "license" / f"license{image_id}.txt"):
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def read_annotations(annotations_dir):
    """Return {label: set(image ids)} for every annotation file.

    'dog.txt' is a potential label and 'dog_r1.txt' a relevant (salient) label.
    """
    labels = {}
    for path in sorted(Path(annotations_dir).glob("*.txt")):
        if path.stem.lower() == "readme":
            continue
        ids = {int(x) for x in _read_lines(path) if x.strip().isdigit()}
        labels[path.stem] = ids
    return labels
