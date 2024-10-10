def multidim2flatIdx(point, shape):
    """
    Compute the flat index for a multi-dimensional point.
    NOTE: The first dimensions are contiguous, rather than in the common order where the last dimensions are contiguous.
    NOTE: Considered order (Col, Row, dim3, dim4, ...), rather than "(dim4, dim3, Row, Col) or (row, col, dim3, dim4)"

    Args:
    - point (list): A list representing the coordinates of the point in each dimension.
    - shape (list): A list representing the shape of the multi-dimensional space.

    Returns:
    - int: The flat index corresponding to the given point.
    """
    flat_index = 0
    multiplier = 1

    # Iterate through each dimension in regular order
    for dim, coord in zip(shape, point):
        flat_index += coord * multiplier
        multiplier *= dim

    return flat_index

# Example usage:
# point = [2, 2, 1]  # Example point coordinates
# shape = [5, 5, 2]  # Example shape of the multi-dimensional space

point = [2, 3, 2, 1]  # Example point coordinates
shape = [4, 5, 3, 2]  # Example shape of the multi-dimensional space

flat_index = multidim2flatIdx(point, shape)
print("Flat index for point", point, "is:", flat_index)
