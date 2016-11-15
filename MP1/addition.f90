program main
  use omp_lib
  implicit none

  real*8 :: start, finish
  real*8, allocatable :: x(:)
  real*8, allocatable :: y(:)
  integer ( kind = 4 ) i
  integer seed
  real*8 arraySum, mySum
  integer num_threads, tid
  integer :: n = 500000000
                
  seed = 13
  call srand(seed)
  
  allocate ( x(1:n) )
  allocate ( y(1:n) )
  
  do i = 1, n
    x(i) = mod(irand(), 1000)
    y(i) = mod(irand(), 1000)
  end do
  arraySum = 0.0
  mySum = 0.0

!----------------------Reduction-----------------------------------! 
!  start = omp_get_wtime()
!  !$omp parallel do reduction(+:arraySum)
!  do i = 1, n
!    arraySum = arraySum +  (x(i)/(y(i)+1))*(y(i)/2.0);
!  end do
!  !$omp end parallel do 
!  finish = omp_get_wtime()
!  print*, "!----------------------Reduction-----------------------------------!" 
!  print *, "Time: ", finish - start, ", Sum: ", arraySum

!-----------------------------------------------------------------! 
!----------------------Critical-----------------------------------! 
!  start = omp_get_wtime()
!  !$omp parallel firstprivate(mySum) 
!  !$omp do
!  do i = 1, n
!    mySum = mySum +  (x(i)/(y(i)+1))*(y(i)/2.0);
!  end do
!  !$omp end do 
!  !$omp critical
!  arraySum = arraySum + mySum
!  !$omp end critical
!  !$omp end parallel
  
!  finish = omp_get_wtime()
!  print*,"!----------------------Critical-----------------------------------!" 
!  print *, "Time: ", finish - start, ", Sum: ", arraySum

!-----------------------------------------------------------------! 
!----------------------Critical and Nowait-----------------------------------! 
  start = omp_get_wtime()
  !$omp parallel firstprivate(mySum) 
  !$omp do
  do i = 1, n
    mySum = mySum +  (x(i)/(y(i)+1))*(y(i)/2.0);
  end do
  !$omp end do NOWAIT
  !$omp critical
  arraySum = arraySum + mySum
  !$omp end critical
  !$omp end parallel
  
  finish = omp_get_wtime()
  print*,"!----------------------Critical and Nowait-----------------------------------!" 
  print *, "Time: ", finish - start, ", Sum: ", arraySum

  deallocate ( x )
  deallocate ( y )
  stop
end


