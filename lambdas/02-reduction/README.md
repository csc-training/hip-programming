# Host-device lambda functions and general kernels

The purpose of this exercise is to understand how the host-device lambda functions work, and how to create a general GPU kernel. Furthermore, differentiating between host and device code paths using ```__HIP_DEVICE_COMPILE__``` macro is demonstrated.

The task is to define two host-device lambda functions that can be passed for the host or the device kernel. Both lambda functions require a single integer argument, and the intended location of these definitions are indicated by `#error`. The first lambda function does not need to capture anything, but must call the predefined function ```helloFromThread(const int i)```. The second lambda function must capture the value of ```pi```, and then must multiply the thread index by the pi, and print this value from each thread.

IMPORTANT NOTE! When using the host-device lambda function with NVIDIA architecures, the following compiler argument must be added for hipcc: `--extended-lambda`
