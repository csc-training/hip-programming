#!/bin/bash

hipcc "--gpu-architecture=sm_70" -O3 Wave_2D.cu -o wgpu

