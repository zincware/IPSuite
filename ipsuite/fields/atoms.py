"""Lazy ASE Atoms loading."""

import pathlib

import h5py
import znh5md
from zntrack import Node
from zntrack.config import (
    NOT_AVAILABLE,
    ZnTrackOptionEnum,
)
from zntrack.fields.base import field

CWD = pathlib.Path(__file__).parent.resolve()


def _frames_getter(self: Node, name: str, suffix: str) -> None:
    with self.state.fs.open((self.nwd / name).with_suffix(suffix), mode="rb") as f:
        with h5py.File(f) as file:
            return znh5md.IO(file_handle=file)[:]


def _frames_save_func(self: Node, name: str, suffix: str) -> None:
    file = (self.nwd / name).with_suffix(suffix)
    io = znh5md.IO(filename=file)
    io.extend(getattr(self, name))


# def Atoms(*, cache: bool = True, independent: bool = False, **kwargs) -> znfields.field:
#     kwargs["metadata"] = kwargs.get("metadata", {})
#     # kwargs["metadata"][ZNTRACK_OPTION] = ZnTrackOptionEnum.OUTS
#     # kwargs["metadata"][ZNTRACK_CACHE] = cache
#     # kwargs["metadata"][ZNTRACK_INDEPENDENT_OUTPUT_TYPE] = independent
#     # kwargs["metadata"][ZNTRACK_FIELD_LOAD] = functools.partial(
#     #     base_getter, func=_frames_getter
#     # )
#     # kwargs["metadata"][ZNTRACK_FIELD_DUMP] = _frames_save_func
#     # kwargs["metadata"][ZNTRACK_FIELD_SUFFIX] = ".h5"
#     return znfields.field(
#         default=NOT_AVAILABLE, getter=plugin_getter, **kwargs, init=False
#     )


def Atoms(*, cache: bool = True, independent: bool = False, **kwargs):
    return field(
        default=NOT_AVAILABLE,
        cache=cache,
        independent=independent,
        zntrack_option=ZnTrackOptionEnum.OUTS,
        dump_fn=_frames_save_func,
        suffix=".h5",
        load_fn=_frames_getter,
        repr=False,
        init=False,
        **kwargs,
    )
