#!/bin/bash
# Extract the prebuilt venv tarball to node-local disk once per node (flock-guarded). Prints venv path.
# Built on node-local /tmp by: uv venv -p /usr/local/bin/python + same packages as venv/, tar of it.
D=/tmp/shawnghu-nfb; mkdir -p $D
( flock 9
  if [ ! -x $D/nfb-venv/bin/python ] || [ /workspace/shawnghu/nanogpt-fb/nfb-venv.tar -nt $D/.stamp ]; then
    rm -rf $D/nfb-venv && tar -xf /workspace/shawnghu/nanogpt-fb/nfb-venv.tar -C $D && touch $D/.stamp
  fi
) 9>$D/.lock
echo $D/nfb-venv
