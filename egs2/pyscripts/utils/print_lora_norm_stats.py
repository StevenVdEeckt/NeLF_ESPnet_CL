import numpy as np

file_name = "temp.txt"
n_pattern = 1  # Set this to the desired pattern count

# Initialize arrays to store sums and counts for each pattern
sums = [np.array([0., 0.]) for _ in range(n_pattern)]
counts = [0 for _ in range(n_pattern)]

with open(file_name, "r", encoding="utf-8") as f:
    line = f.readline()
    line_num = 0
    while line:
        line = line.split("tensor([")[-1]
        line = line.strip("])\n")
        line = line.split(", ")
        x, y = float(line[0]), float(line[1])
        
        pattern_index = line_num % n_pattern
        sums[pattern_index] += np.array([x, y])
        counts[pattern_index] += 1
        
        line_num += 1
        line = f.readline()

# Calculate averages
averages = [sums[i] / max(counts[i], 1) for i in range(n_pattern)]

# Print results
for i in range(n_pattern):
    print(f"({counts[i]}) Average weights for pattern {i+1}: {averages[i]}")


