#slicing: https://www.w3schools.com/python/python_strings_slicing.asp

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Your data
batch_size = 3

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    print("Batch:", batch)
