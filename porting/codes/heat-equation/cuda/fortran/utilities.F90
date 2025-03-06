! Utility routines for heat equation solver
!   NOTE: This file does not need to be edited!
module utilities
  use heat

contains

  ! Copy the data from one field to another
  ! Arguments:
  !   from_field (type(field)): variable to copy from
  !   to_field (type(field)): variable to copy to
  subroutine copy_fields(from_field, to_field)

    implicit none

    type(field), intent(in) :: from_field
    type(field), intent(out) :: to_field

    ! Consistency checks
    if (.not.allocated(from_field%data)) then
       write (*,*) "Can not copy from a field without allocated data"
       stop
    end if
    if (.not.allocated(to_field%data)) then
       ! Target is not initialize, allocate memory
       allocate(to_field%data(lbound(from_field%data, 1):ubound(from_field%data, 1), &
            & lbound(from_field%data, 2):ubound(from_field%data, 2)))
    else if (any(shape(from_field%data) /= shape(to_field%data))) then
       write (*,*) "Wrong field data sizes in copy routine"
       print *, shape(from_field%data), shape(to_field%data)
       stop
    end if

    to_field%data = from_field%data

    to_field%nx = from_field%nx
    to_field%ny = from_field%ny
    to_field%nx_full = from_field%nx_full
    to_field%ny_full = from_field%ny_full
    to_field%dx = from_field%dx
    to_field%dy = from_field%dy
  end subroutine copy_fields

  function average(field0) 
    use mpi

    implicit none

    real(dp) :: average
    type(field) :: field0

    real(dp) :: local_average
    integer :: rc

    local_average = sum(field0%data(1:field0%nx, 1:field0%ny))
    call mpi_allreduce(local_average, average, 1, MPI_DOUBLE_PRECISION, MPI_SUM,  &
               &       MPI_COMM_WORLD, rc)
    average = average / (field0%nx_full * field0%ny_full)
    
  end function average

end module utilities
