# Kernel: saxpy

Write a device kernel that calculates the single precision BLAS operation
**saxpy**, i.e. `y = a * x + y`.

- Initialise the vectors `x` and `y` with some values on the CPU
- Perform the computation on the host to generate reference values
- Allocate memory on the device for `x` and `y`
- Copy the host `x` to device `x`, and host `y` to device `y`
- Perform the computation on the device
- Copy the device `y` back to the host `y`
- Confirm the correctness: Is the host computed `y` equal to the device computed `y`?

You may start from a skeleton code provided in [saxpy.cpp](saxpy.cpp).
