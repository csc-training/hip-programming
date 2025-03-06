#!/bin/bash

module load pgi

pgf90 -O3 -ta=tesla:cc70 -o saxpy main.cuf 
