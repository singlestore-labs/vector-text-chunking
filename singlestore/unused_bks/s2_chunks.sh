#!/bin/bash
# Wrapper script for SingleStore chunks with default password

# Set your default password here
DEFAULT_PASSWORD="root"

python singlestore_chunks.py "$@" --password "$DEFAULT_PASSWORD"