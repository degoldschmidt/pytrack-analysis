#!/bin/sh
cd ..
sudo python3 setup.py install --force > /dev/null #2>&1
cd scripts

BASEDIR=$1
OPTION=$2
FORCE=$3

if [ "$2" = "all" ] || [ "$2" = "posttracking" ] || [ "$2" = "registration" ]; then
    python3 run_posttracking.py "$1" --option "$2""$3"
fi

if [ "$2" = "manual_geometry" ]; then
    python3 run_manual_geometry.py "$1"
fi

if [ "$2" = "all" ] || [ "$2" = "kinematics" ]; then
    python3 run_kinematics.py "$1" "$3" "$4"
fi

if [ "$2" = "all" ] || [ "$2" = "classify" ]; then
    python3 run_classification.py "$1" "$3" "$4"
fi

if [ "$2" = "all" ] || [ "$2" = "segments" ]; then
    python3 run_segmentation.py "$1" "$3" "$4"
fi
