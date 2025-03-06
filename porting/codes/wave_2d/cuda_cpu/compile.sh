#!/bin/bash
g++ -O3 Wave_2D.c -o wcpu

nvcc -arch=sm_70 -O3 Wave_2D.cu -o wgpu

