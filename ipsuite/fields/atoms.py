"""Lazy ASE Atoms loading."""

import pathlib

import typing_extensions as tyex
import znh5md
import zntrack

from ipsuite.utils.helpers import make_hdf5_file_opener

CWD = pathlib.Path(__file__).parent.resolve()


def _frames_getter(self: zntrack.Node, name: str, suffix: str) -> znh5md.IO:
    file_factory = make_hdf5_file_opener(self, (self.nwd / name).with_suffix(suffix))
    return znh5md.IO(file_factory=file_factory)


def _frames_save_func(self: zntrack.Node, name: str, suffix: str) -> None:
    file = (self.nwd / name).with_suffix(suffix)
    io = znh5md.IO(filename=file)
    io.create_file()
    io.extend(getattr(self, name))


@tyex.deprecated(
    "use explicit frames_path: Path ="
    " zntrack.outs_path(zntrack.nwd / 'frames.h5')"
    " with property frames instead."
)
def Atoms(*, cache: bool = True, independent: bool = False, **kwargs):
    return zntrack.field(
        default=zntrack.NOT_AVAILABLE,
        cache=cache,
        independent=independent,
        field_type=zntrack.FieldTypes.OUTS,
        dump_fn=_frames_save_func,
        suffix=".h5",
        load_fn=_frames_getter,
        repr=False,
        init=False,
        **kwargs,
    )
