#!/usr/bin/env bash
# Wrap index.html (artifact form, no document skeleton) into .preview.html for local checks.
cd "$(dirname "$0")/.."
{ echo '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'; cat index.html; echo '</body></html>'; } > .preview.html
