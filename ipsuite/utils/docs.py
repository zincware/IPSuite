import subprocess
import tempfile

import zntrack


def create_dvc_git_env_for_doctest() -> tempfile.TemporaryDirectory:
    """Create a temporary directory and initilize git
    and DVC in the new directory.

    This is necessary to construct IPS workflows in
    the Examples part of node docstrings.
    Call this function from the testsetup sphinx
    directive.

    After testing the docstring in the temp directory,
    use tmp_path.cleanup() to delete the directory.
    Do this from the testcleanup directive

    Returns
    -------
    tempfile.TemporaryDirectory
        returns a temporary Directory with git and DVC initalized.
    """

    tmp_path = zntrack.utils.cwd_temp_dir()
    _ = subprocess.run(["git", "init"])
    _ = subprocess.run(["dvc", "init"])
    return tmp_path
