#!/usr/bin/env bash

if [ ! -d "init-script" ]; then
    echo "must run from project root"
    exit 1
fi

git submodule update --init --recursive
cd external/WANDbl && ./build.sh && cd -
mkdir -p external/local/bin
cp external/WANDbl/bin/* external/local/bin
mkdir -p build
cd build && cmake .. && make && cd -
cp $(find build -maxdepth 1 -type f -executable | xargs) external/local/bin
