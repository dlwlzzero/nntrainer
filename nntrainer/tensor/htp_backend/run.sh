#!/bin/bash

adb_dir="/usr/lib/android-sdk/platform-tools/adb"

# Check if target directory is provided as argument
if [ $# -eq 0 ]; then
    target_dir="/data/local/tmp/htp_backend"
    echo "No target directory provided. Using default: $target_dir"
else
    target_dir="$1"
fi
echo ""

set -e

echo "ADB push files to device: $target_dir"
${adb_dir} push build_htp/libhtp_ops.so "$target_dir"
${adb_dir} push build_htp/libhtp_ops_skel.so "$target_dir"
${adb_dir} push build_htp/htp_ops_test "$target_dir"
echo ""

echo "Execute htp_ops_test on device"
${adb_dir} shell "cd $target_dir && export LD_LIBRARY_PATH=. && ./htp_ops_test"
echo ""
