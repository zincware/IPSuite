import zntrack
import subprocess
import os

def create_dvc_git_env_for_doctest():
    # Generate a temp directory and initalize dvc and git in it (also delete it up after that)
    tmp_path = zntrack.utils.cwd_temp_dir()
    _ = subprocess.run(["git", "init"]) 
    _ = subprocess.run(["dvc", "init"])
    yield 

    tmp_path.cleanup()
    yield
