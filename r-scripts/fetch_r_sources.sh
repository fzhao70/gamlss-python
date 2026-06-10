#!/usr/bin/env bash
# Download the original R package sources this port was transcribed from.
# They are kept out of the git repository (see .gitignore) but are useful
# when reviewing the port against the R implementation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p r-src
cd r-src
BASE=https://cran.r-project.org/src/contrib
# exact versions the port is verified against; if archived, fall back to
# the CRAN Archive URL
for p in gamlss_5.5-0 gamlss.dist_6.1-1 gamlss.data_6.0-7; do
  name=${p%%_*}
  curl -fsSLO "$BASE/$p.tar.gz" \
    || curl -fsSLO "$BASE/Archive/$name/$p.tar.gz"
  tar xzf "$p.tar.gz"
done
echo "R sources extracted into r-src/"
