program rand_test
  use iso_c_binding
  use iso_fortran_env, only : INT64
  use hipfort
  use hipfort_check
  use hipfort_hiprand

  implicit none

  interface
     subroutine launch(x_d, y_d, inside_d, N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr), value :: x_d, y_d, inside_d
       integer(c_int64_t), value :: N  ! Ensure use of correct C type for INT64
     end subroutine
  end interface

  integer(c_int64_t) :: nsamples
  character(len=85) :: arg
  real :: pi1, pi2
  integer(c_size_t) :: Nbytes, Sbytes

  if (command_argument_count() /= 1) then
    STOP 'Usage: pi N where N is the number of samples'
  end if

  call get_command_argument(1, arg)
  read(arg, *) nsamples

  pi1 = cpu_pi(nsamples)
  write(*,*) 'Pi calculated with CPU', pi1
  pi2 = gpu_pi(nsamples)
  write(*,*) 'Pi calculated with GPU', pi2

contains

  real function cpu_pi(n)
    implicit none
    integer(c_int64_t) :: n
    integer :: i, inside

    real, allocatable :: x(:), y(:)

    allocate(x(1:n))
    allocate(y(1:n))

    call random_number(x)
    call random_number(y)

    inside = 0
    do i = 1, n
      if (x(i)**2 + y(i)**2 < 1.0) then
        inside = inside + 1
      end if
    end do

    cpu_pi = 4.0 * real(inside) / real(n)

  end function cpu_pi

  real function gpu_pi(n)
    use hipfort
    use hipfort_check
    use hipfort_hiprand
    implicit none
    integer(c_int64_t) :: n
    integer :: inside
    type(c_ptr) :: gen = c_null_ptr
    type(c_ptr) :: x_d, y_d, inside_d
    real(c_float), allocatable, target :: x(:), y(:)
    integer(c_size_t) :: istat

    allocate(x(1:n))
    allocate(y(1:n))
    Nbytes = sizeof(x)

    call hipCheck(hipMalloc(x_d, Nbytes))
    call hipCheck(hipMalloc(y_d, Nbytes))

    istat = hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_DEFAULT)

    istat = hiprandGenerateUniform(gen, x_d, n)
    istat = hiprandGenerateUniform(gen, y_d, n)

    inside = 0
    Sbytes = sizeof(inside)
    call hipCheck(hipMalloc(inside_d, Sbytes))
    call hipCheck(hipMemcpy(inside_d, c_loc(inside), Sbytes, hipMemcpyHostToDevice))

    call launch(x_d, y_d, inside_d, n)

    call hipCheck(hipMemcpy(c_loc(inside), inside_d, Sbytes, hipMemcpyDeviceToHost))

    gpu_pi = 4.0 * real(inside) / real(n)

    deallocate(x, y)
  end function gpu_pi

end program rand_test
