#!/bin/bash

# Vector sizes and thread counts to test
VECTOR_SIZES=("10000" "1000000" "100000000")
THREAD_COUNTS=("4" "8" "16")

# Run tests
for size in "${VECTOR_SIZES[@]}"; do
    echo -e "\n========== Vector Size: $size =========="
    for threads in "${THREAD_COUNTS[@]}"; do
        echo -e "\nRunning with $threads threads..."
        ./saxpy "$size" "$threads"
    done
done

echo -e "\nAll tests completed!"
