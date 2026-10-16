#!/bin/bash

# File paths
mp4_file="unique_mp4_files.txt"
eaf_file="unique_eaf_files.txt"

# Remove extensions and sort the files
sort <(sed 's/\.mp4$//' "$mp4_file") > mp4_files_clean.txt
sort <(sed 's/\.eaf$//' "$eaf_file") > eaf_files_clean.txt

# Use comm to find eaf files with no corresponding mp4 files
comm -23 eaf_files_clean.txt mp4_files_clean.txt > eaf_files_without_mp4.txt

# Display the result
cat eaf_files_without_mp4.txt

