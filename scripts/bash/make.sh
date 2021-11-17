#!/bin/bash

if [ -d "./build" ]; then
	cd ./build
else
	mkdir ./build
	cd ./build
	cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -Wno-dev
fi

N=$(nproc)
if [ $N -gt 4 ]; then
	if [ $N -gt 8 ]; then
		make -j8
	else
		make -j4
	fi
else
	make
fi