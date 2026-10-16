import csv
import pandas as pd

def read_file(file_name, first_value=True):
    lora_A = {}
    lora_B = {}
    with open(file_name, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines) - 1:
            layer_line = lines[i].strip()
            weights_line = lines[i + 1].strip()

            if "for layer" not in layer_line or "weights = tensor([" not in weights_line:
                i += 1
                continue

            layer_name = layer_line.split("for layer ")[1]
            try:
                weights = weights_line.split("weights = tensor([")[1]
            except Exception as e:
                raise Exception(f"{e} \n (layer_line = {layer_line}, \n weights_line = {weights_line})")
            weights = weights.strip('])')
            weightsA = list(map(float, weights.split(", ")))

            if first_value:
                valueA = weightsA[0]
            else:
                valueA = weightsA[1]

            if layer_name in lora_A:
                lora_A[layer_name].append(valueA)
            else:
                lora_A[layer_name] = [valueA]

            i += 2
    
    # Calculate the average for each layer
    avg_lora_A = {layer: sum(values) / len(values) for layer, values in lora_A.items()}
    
    return avg_lora_A

def main():
    file1 = "temp1.txt"
    file2 = "temp2.txt"

    lora_A1 = read_file(file1, first_value=True)
    lora_A2 = read_file(file2, first_value=False)

    # Prepare data for CSV
    data_A = []

    layers = set(lora_A1.keys()).union(set(lora_A2.keys()))
    
    for layer in layers:
        row_A = [layer, lora_A1.get(layer, 'N/A'), lora_A2.get(layer, 'N/A')]
        data_A.append(row_A)
    
    df_A = pd.DataFrame(data_A, columns=['Layer', 'Task 1', 'Task 2'])
    
    # Write to CSV with two sheets
    with pd.ExcelWriter('weights_comparison.xlsx') as writer:
        df_A.to_excel(writer, sheet_name='LoRA-A', index=False)

if __name__ == "__main__":
    main()

