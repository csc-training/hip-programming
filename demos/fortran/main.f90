program test_saxpy

    use iso_c_binding
    implicit none

    integer, parameter :: n = 1000000

    real(c_float), allocatable, target :: x(:), y(:)

    type(c_ptr) :: x_d
    type(c_ptr) :: y_d

    integer(c_size_t) :: nbytes
    integer(c_int) :: ierr

    real(c_float) :: a

    integer :: i

    ! ==========================================
    ! HIP memcpy kinds
    ! ==========================================

    integer(c_int), parameter :: hipMemcpyHostToDevice = 1
    integer(c_int), parameter :: hipMemcpyDeviceToHost = 2

    ! ==========================================
    ! INTERFACES
    ! ==========================================

    interface

        function hipMalloc(ptr, bytes) bind(C, name="hipMalloc")
            use iso_c_binding
            type(c_ptr) :: ptr
            integer(c_size_t), value :: bytes
            integer(c_int) :: hipMalloc
        end function

        function hipFree(ptr) bind(C, name="hipFree")
            use iso_c_binding
            type(c_ptr), value :: ptr
            integer(c_int) :: hipFree
        end function

        function hipMemcpy(dst, src, bytes, kind) bind(C, name="hipMemcpy")
            use iso_c_binding
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: bytes
            integer(c_int), value :: kind
            integer(c_int) :: hipMemcpy
        end function

        subroutine saxpy(n, a, x_d, y_d) bind(C, name="saxpy_")
            use iso_c_binding

            integer(c_int), value :: n
            real(c_float), value :: a

            type(c_ptr), value :: x_d
            type(c_ptr), value :: y_d

        end subroutine

    end interface

    allocate(x(n), y(n))

    x = 1.0_c_float
    y = 2.0_c_float

    a = 3.0_c_float

    nbytes = n * c_sizeof(x(1))

    ierr = hipMalloc(x_d, nbytes)
    ierr = hipMalloc(y_d, nbytes)

    ierr = hipMemcpy(x_d, c_loc(x), nbytes, hipMemcpyHostToDevice)
    ierr = hipMemcpy(y_d, c_loc(y), nbytes, hipMemcpyHostToDevice)

    call saxpy(n, a, x_d, y_d)

    ierr = hipMemcpy(c_loc(y), y_d, nbytes, hipMemcpyDeviceToHost)

    print *, "y(1) =", y(1)
    print *, "expected =", a * 1.0 + 2.0

    ierr = hipFree(x_d)
    ierr = hipFree(y_d)

    deallocate(x, y)

end program
