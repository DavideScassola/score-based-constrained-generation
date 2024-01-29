#!/bin/bash

for file in "$1"/*
do
    #echo "$file"
    python main-generate.py "$file"
done