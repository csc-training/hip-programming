# Creating a stream and launching a GPU kernel on it

This exercise demonstrates how a HIP stream is created, and how
memory transfers and a GPU kernel can be submitted using it.

If your program executes correctly, you should get the following output:

```
1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 1.000000 
error: 0.000000
```

Printing the 10 first values in your array and the maximum error across all elements.

## Instructions

In this exercise, you will:

- create a HIP stream
- perform asynchronous memory transfers using the stream
- launch a kernel into the stream
- synchronize the host with the stream
- destroy the stream

The GPU kernel and a correctness checking function are already implemented.

## HIP functions used

The following HIP functions are needed in this exercise:

* `hipStreamCreate()`
* `hipMemcpyAsync()`
* `hipStreamSynchronize()`
* `hipStreamDestroy()`

<details>
<summary><strong>Additional exercise: Async Malloc and Free</strong></summary>

Two other API calls that are normally blocking, can also be submitted to the stream.
For these, use:

* `hipMallocAsync()`
* `hipFreeAsync()`

However, this is not strictly necessary in this exercise.

</details>

## Bonus: What does the kernel compute exactly?

<details>
<summary><strong>Bonus: What does the GPU kernel in this exercise compute?</strong></summary>

In the main function, we set all elements in our array `a` to zero first.

We launch the kernel, where each GPU thread computes one element in the array `a`.
The kernel evaluates the trigonometric identity:

$$
\sin^2(x) + \cos^2(x) = 1
$$

for each thread id, mimicking a workload for element-wise arithmetic operations.

The function `max_error()` checks each array entry against the expected value `1.0f`,
and returns the largest error.

</details>