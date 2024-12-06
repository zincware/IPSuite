#!/bin/bash
set -e

# Source any necessary environment configuration
source /opt/tools/cp2k/tools/toolchain/install/setup

# Check if arguments were passed; if not, open /bin/bash
if [ "$#" -eq 0 ]; then
    exec /bin/bash
else
    # Run the command passed to the container
    exec "$@"
fi
