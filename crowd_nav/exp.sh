#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec python -m crowd_nav.exp "$@"
