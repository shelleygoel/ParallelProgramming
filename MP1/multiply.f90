program main
  use omp_lib
  implicit none

  real*8 :: start, finish
  integer ( kind = 4 ) i, j, k
  integer seed
  real*8 arraySum
  real*4 temp1
  integer num_threads, tid
  integer :: n = 1000
  integer(kind=4) :: m,l   !unroll and jam factor
  real*4, DIMENSION(:), allocatable :: temp
  real*4, DIMENSION(:,:), allocatable :: A, B, C
                

  allocate( A(n,n) )
  allocate( B(n,n) )
  allocate( C(n,n) )
  seed = 13
  call srand(seed)
  
  do j = 1, n
    do i = 1, n
      A(i,j) = mod(irand(), 10)
      B(i,j) = mod(irand(), 10)
    end do
  end do
  C = 0

!---------------------parallelizing column computation with unroll and jam 2-------------!  
  m = 50
  start = omp_get_wtime()
  !$omp parallel do private(i,k,l)
  do j = 1, n
    do i = 1,n,m
      do k = 1, n
         do l = 0,m-1
            C(i+l,j) = C(i+l,j) + (A(i+l,k)*B(k,j))
         end do
         !C(i+1,j) = C(i+1,j) + (A(i+1,k)*B(k,j))
         !C(i+2,j) = C(i+2,j) + (A(i+2,k)*B(k,j))
         !C(i+3,j) = C(i+3,j) + (A(i+3,k)*B(k,j))
         !C(i+4,j) = C(i+4,j) + (A(i+4,k)*B(k,j))
         !C(i+5,j) = C(i+5,j) + (A(i+5,k)*B(k,j))
         !C(i+6,j) = C(i+6,j) + (A(i+6,k)*B(k,j))
         !C(i+7,j) = C(i+7,j) + (A(i+7,k)*B(k,j))
         !C(i+8,j) = C(i+8,j) + (A(i+8,k)*B(k,j))
         !C(i+9,j) = C(i+9,j) + (A(i+9,k)*B(k,j))
      end do
    end do
  end do
  !$omp end parallel do 
  finish = omp_get_wtime()
  
  arraySum = 0.0
  do j = 1, n
    do i = 1, n
      arraySum = arraySum + C(i,j)
    end do
  end do
  print*, "!---------------------parallelizing column computation with unroll and jam-------------!"  
  print *, "Time: ", finish - start, ", Sum: ", arraySum

!---------------------parallelizing row computation-------------!  
!  start = omp_get_wtime()
!  !$omp parallel do private(i,k,temp1)
!  do i = 1, n
!    do j = 1, n
!      temp1 = 0
!      do k = 1, n
!        temp1 = temp1 + (A(i,k)*B(k,j))
!      end do
!      C(i,j) = temp1
!    end do
!  end do
!  !$omp end parallel do 
!  finish = omp_get_wtime()
  
!  arraySum = 0.0
!  do j = 1, n
!    do i = 1, n
!      arraySum = arraySum + C(i,j)
!    end do
!  end do
!  print*, "!---------------------parallelizing row computation-------------!"  
!  print *, "Time: ", finish - start, ", Sum: ", arraySum

!---------------------parallelizing column computation-------------!  
!  start = omp_get_wtime()
!  !$omp parallel do private(i,k,l,temp1)
!  do j = 1, n
!    do i = 1,n
!      temp1 = 0
!      do k = 1, n
!         temp1 = temp1 + (A(i,k)*B(k,j))
!      end do
!      C(i,j) = temp1
!    end do
!  end do
!  !$omp end parallel do 
!  finish = omp_get_wtime()
  
!  arraySum = 0.0
!  do j = 1, n
!    do i = 1, n
!      arraySum = arraySum + C(i,j)
!    end do
!  end do
!  print*, "!---------------------parallelizing column computation-------------!"  
!  print *, "Time: ", finish - start, ", Sum: ", arraySum

  deallocate( A )
  deallocate( B )
  deallocate( C )
  stop
end


