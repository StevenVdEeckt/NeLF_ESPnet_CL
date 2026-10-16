def dhont_allocation(scores, total_spots, threshold=0.0, multiplicators=None):
    """
    Implements the Dhont method for allocating a fixed number of spots based on floating point scores,
    with the option to allocate multiple spots at a time for specific entries.

    Parameters:
    scores (dict): Dictionary with keys as names and values as floating point scores.
    total_spots (int): Total number of spots to be allocated.
    threshold (float): Minimum score required to be considered for spot allocation.
    multiplicators (dict): Dictionary with keys as names and values as the multiplicators for allocating spots.
                           If a name is not present, it defaults to 1.

    Returns:
    dict: Dictionary with keys as names and values as the number of allocated spots.
    """
    
    if multiplicators is None:
        multiplicators = {}

    # Filter out scores below the threshold
    filtered_scores = {name: score for name, score in scores.items() if score > threshold}

    # Initialize allocation dictionary with zero spots for each entry
    allocation = {name: 0 for name in filtered_scores}

    # Generate a list of tuples (name, score) for computation
    score_list = [(name, score) for name, score in filtered_scores.items()]

    # Allocate spots
    for _ in range(total_spots):
        # Compute the current highest averages
        highest_average = 0
        highest_name = None
        
        for name, score in score_list:
            multiplicator = multiplicators.get(name, 1)
            current_average = score / (allocation[name] // multiplicator + 1)
            if current_average > highest_average:
                highest_average = current_average
                highest_name = name
        
        # Allocate the multiplicator number of spots to the name with the highest average
        allocation[highest_name] += multiplicators.get(highest_name, 1)
    
    return allocation


def print_dict(d):
    for key, values in d.items():
        print(f"{key}: {values}")

# Example usage:
scores = {
    "Encoder.0": 0.00,
    "Encoder.1": 1.07,
    "Encoder.2": 1.56,
    "Encoder.3": 1.53,
    "Encoder.4": 1.36,
    "Encoder.5": 1.97,
    "Decoder.0": 1.07,
    "Decoder.1": 1.16,
    "Decoder.2": 1.21,
    "Decoder.3": 1.67,
    "Decoder.4": 1.53,
    "Decoder.5": 1.64,
}
total_spots = 100
threshold = 0.0
dimension = 8
multiplicators = {
    "Encoder.0": 2,
    "Encoder.1": 2,
    "Encoder.2": 2,
    "Encoder.3": 2,
    "Encoder.4": 2,
    "Encoder.5": 2,
}

allocation = dhont_allocation(scores, total_spots, threshold, multiplicators=multiplicators)

print("Allocation of spots: ")
print_dict(allocation)

adapters = {k: (int(v * dimension / 2) if k.startswith('Encoder') else v * dimension) for k, v in allocation.items()}
print("Adapter dimension per block: ")
print_dict(adapters)
