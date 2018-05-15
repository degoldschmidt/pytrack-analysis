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

#if [ "$2" = "all" ] || [ "$2" = "posttracking" ]; then
#    python3 run_database.py "$1" --option "$2""$3"
#fi
