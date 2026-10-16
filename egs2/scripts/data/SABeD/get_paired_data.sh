#!/bin/bash

# File paths
mp4_file="unique_mp4_files.txt"
eaf_file="unique_eaf_files.txt"

# Remove extensions and sort the files
sort <(sed 's/\.mp4$//' "$mp4_file") > mp4_files_clean.txt
sort <(sed 's/\.eaf$//' "$eaf_file") > eaf_files_clean.txt

# Use comm to find common file names
comm -12 mp4_files_clean.txt eaf_files_clean.txt > paired_data.txt

# Display the result
cat paired_data.txt
wc -l paired_data.txt

