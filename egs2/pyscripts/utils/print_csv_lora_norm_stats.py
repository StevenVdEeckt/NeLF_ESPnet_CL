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

            if "for layer" not in layer_line or "weightsA = tensor([" not in weights_line:
                i += 1
                continue

            layer_name = layer_line.split("for layer ")[1]
            try:
                weights = weights_line.split("weightsA = tensor([")[1].split("]) - weightsB = tensor([")
            except Exception as e:
                raise Exception(f"{e} \n (layer_line = {layer_line}, \n weights_line = {weights_line})")
            weights[-1] = weights[-1].strip('])')
            weightsA = list(map(float, weights[0].split(", ")))
            weightsB = list(map(float, weights[1].split(", ")))

            if first_value:
                valueA = weightsA[0]
                valueB = weightsB[0]
            else:
                valueA = weightsA[1]
                valueB = weightsB[1]

            if layer_name in lora_A:
                lora_A[layer_name].append(valueA)
                lora_B[layer_name].append(valueB)
            else:
                lora_A[layer_name] = [valueA]
                lora_B[layer_name] = [valueB]

            i += 2
    
    # Calculate the average for each layer
    avg_lora_A = {layer: sum(values) / len(values) for layer, values in lora_A.items()}
    avg_lora_B = {layer: sum(values) / len(values) for layer, values in lora_B.items()}
    
    return avg_lora_A, avg_lora_B

def main():
    file1 = "temp1.txt"
    file2 = "temp2.txt"

    lora_A1, lora_B1 = read_file(file1, first_value=True)
    lora_A2, lora_B2 = read_file(file2, first_value=False)

    # Prepare data for CSV
    data_A = []
    data_B = []

    layers = set(lora_A1.keys()).union(set(lora_A2.keys()))
    
    for layer in layers:
        row_A = [layer, lora_A1.get(layer, 'N/A'), lora_A2.get(layer, 'N/A')]
        row_B = [layer, lora_B1.get(layer, 'N/A'), lora_B2.get(layer, 'N/A')]
        data_A.append(row_A)
        data_B.append(row_B)
    
    df_A = pd.DataFrame(data_A, columns=['Layer', 'Task 1', 'Task 2'])
    df_B = pd.DataFrame(data_B, columns=['Layer', 'Task 1', 'Task 2'])
    
    # Write to CSV with two sheets
    with pd.ExcelWriter('weights_comparison.xlsx') as writer:
        df_A.to_excel(writer, sheet_name='LoRA-A', index=False)
        df_B.to_excel(writer, sheet_name='LoRA-B', index=False)

if __name__ == "__main__":
    main()

