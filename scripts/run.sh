#!/bin/sh
cd ..
sudo python3 setup.py install --force > /dev/null #2>&1
cd scripts

BASEDIR=$1
OPTION=$2
FORCE=$3

if [ "$2" = "all" ] || [ "$2" = "post_tracking" ] || [ "$2" = "registration" ] || [ "$2" = "jump_detection" ]; then
    python3 run_posttracking.py "$1" --option "$2""$3"
fi

if [ "$2" = "manual_geometry" ]; then
    python3 run_manual_geometry.py "$1"
fi

if [ "$2" = "all" ] || [ "$2" = "kinematics" ]; then
    if [ "$3" = "overwrite" ]; then
      python3 run_kinematics.py "$1" --overwrite
    else
      python3 run_kinematics.py "$1"
    fi
fi

if [ "$2" = "all" ] || [ "$2" = "classifier" ]; then
    if [ "$3" = "overwrite" ]; then
      python3 run_classification.py "$1" --overwrite
    else
      python3 run_classification.py "$1"
    fi
fi

if [ "$2" = "all" ] || [ "$2" = "segments" ]; then
    if [ "$3" = "overwrite" ]; then
      python3 run_segmentation.py "$1" --overwrite
    else
      python3 run_segmentation.py "$1"
    fi
fi

if [ "$2" = "all" ] || [ "$2" = "stats" ]; then
    if [ "$3" = "overwrite" ]; then
      python3 run_stats.py "$1" --overwrite
    else
      python3 run_stats.py "$1"
    fi
fi
