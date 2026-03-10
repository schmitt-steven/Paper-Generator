# Paper Specification

## General Information

### Topic
Efficient Sorting Algorithms

### Hypothesis
A new hybrid sorting algorithm combines the best of quicksort and mergesort.

## Section Requirements

### Abstract
We present HybridSort, a novel algorithm that combines the pivot strategy of quicksort with the guaranteed worst-case performance of mergesort.

### Introduction
Sorting is an important primitive in computer science. Standard algorithms often struggle with either worst-case time complexity or memory overhead. We introduce HybridSort to address these limitations.

### Related Work
Standard quicksort and mergesort are the baseline algorithms. We review their time and space complexity tradeoffs.

### Methods
The algorithm operates by dividing the array using a median-of-three pivot until a threshold is reached, after which it applies a bottom-up merge strategy.

### Results
Performance is evaluated by measuring the wall-clock execution time on arrays of various sizes, comparing HybridSort against standard implementations.

### Discussion
HybridSort mitigates the deepest recursion trees of quicksort. The primary limitation is the increased code complexity.

### Conclusion
HybridSort is an efficient general-purpose sorting algorithm. It handles worst-case scenarios gracefully.

### Acknowledgements