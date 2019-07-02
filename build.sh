#!/bin/bash

mkdir -p ff/build
cd ff/build
pushd .
cmake -DMULTICORE=ON ..
make -j12 ff
cd ../..
make play 
