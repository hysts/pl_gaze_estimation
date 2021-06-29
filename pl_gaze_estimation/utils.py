import pathlib


def str2path(path: str) -> pathlib.Path:
    new_path = pathlib.Path(path)
    if new_path.as_posix().startswith('~'):
        new_path = new_path.expanduser()
    return new_path
