"""Lazy ASE Atoms loading."""

import pathlib

import h5py
import znh5md
import zntrack

CWD = pathlib.Path(__file__).parent.resolve()


def _frames_getter(self: zntrack.Node, name: str, suffix: str) -> None:
    with self.state.fs.open((self.nwd / name).with_suffix(suffix), mode="rb") as f:
        with h5py.File(f) as file:
            return znh5md.IO(file_handle=file)[:]


def _frames_save_func(self: zntrack.Node, name: str, suffix: str) -> None:
    file = (self.nwd / name).with_suffix(suffix)
    io = znh5md.IO(filename=file)
    io.create_file()
    io.extend(getattr(self, name))


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
