#!/bin/bash

# Check the command passed to the container
case "$1" in
  python)
    # If "python" is passed, open a Python shell
    exec python
    ;;
  dvc)
    # If "dvc" is passed, run "dvc repro" with the rest of the arguments
    shift
    exec dvc "$@"
    ;;
  zntrack)
    # If "zntrack" is passed, run "zntrack" with the rest of the arguments
    shift
    exec zntrack "$@"
    ;;
  *)
    # Default: run the command as is
    exec "$@"
    ;;
esac
