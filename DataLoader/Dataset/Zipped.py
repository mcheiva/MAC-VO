"""
Read-only view of a directory stored inside one or more zip archives, so datasets can be read in
place from their distribution zips (e.g. the official TartanAir v1 <env>/<level>/<modality>.zip).

Archives are opened lazily once per process: instances hold only paths and member names, so they
pickle cleanly into multiprocessing pools and DataLoader workers.
"""
import io
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np


class ZipMember:
    """A file inside a zip archive; quacks like the parts of `Path` the dataset loaders use."""
    __slots__ = ("archive", "member", "name")

    def __init__(self, archive: Path, member: str):
        self.archive = archive
        self.member  = member
        self.name    = member.rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str: return os.path.splitext(self.name)[1]

    def __lt__(self, other: "ZipMember") -> bool: return self.member < other.member
    def __str__(self) -> str: return f"{self.archive}::{self.member}"

    def read_bytes(self) -> bytes: return ZipDir.open_archive(self.archive).read(self.member)


class ZipDir:
    """
    The directory `directory` (e.g. "ocean/Easy/P000/flow") merged across `archives`. Only direct
    children are listed. `zip_dir / name` returns the member, like `Path / name`.
    """
    _archives: dict[tuple[int, Path], zipfile.ZipFile] = {}

    def __init__(self, archives: list[Path], directory: str):
        self.archives  = archives
        self.directory = directory.strip("/")
        prefix = self.directory + "/" if self.directory else ""
        self.members: dict[str, ZipMember] = dict()
        for archive in archives:
            for member in self.open_archive(archive).namelist():
                name = member[len(prefix):]
                if member.startswith(prefix) and name and "/" not in name:
                    self.members[name] = ZipMember(archive, member)

    @classmethod
    def open_archive(cls, archive: Path) -> zipfile.ZipFile:
        key = (os.getpid(), archive)
        if key not in cls._archives:
            cls._archives[key] = zipfile.ZipFile(archive)
        return cls._archives[key]

    def exists(self) -> bool: return len(self.members) > 0
    def iterdir(self):        return iter(self.members.values())
    def __truediv__(self, name: str) -> ZipMember: return self.members[name]
    def __str__(self) -> str: return "+".join(str(a) for a in self.archives) + "::" + self.directory


def imread(path: Path | ZipMember, flags: int) -> np.ndarray | None:
    if isinstance(path, ZipMember):
        return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)
    return cv2.imread(str(path), flags)


def npload(path: Path | ZipMember) -> np.ndarray:
    if isinstance(path, ZipMember):
        return np.load(io.BytesIO(path.read_bytes()), allow_pickle=False)
    return np.load(str(path))


def loadtxt(path: Path | ZipMember) -> np.ndarray:
    if isinstance(path, ZipMember):
        return np.loadtxt(io.StringIO(path.read_bytes().decode()))
    return np.loadtxt(str(path))
