program main
  use omp_lib
  implicit none

  real*8 :: start, finish
  integer ( kind = 4 ) i, j, k
  integer seed
  real*8 arraySum, cnt
  real*4 temp,temp2
  integer num_threads, tid
  integer :: n = 1000
  integer :: jj, kk,ii, jstart, kstart,kend,jend, b
  real*4, DIMENSION(:,:,:), allocatable :: A
                
  allocate( A(n,n,n) )
  seed = 13
  call srand(seed)
  
  do k = 1, n
    do j = 1, n
      do i = 1, n
        A(i,j,k) = mod(irand(), 10)
      end do
    end do
  end do
!--------------parallelizing k loop with unrolling and jamming i------------!
  b = 10; cnt = 0
  start = omp_get_wtime()
  !$omp parallel do private(i,j,ii,temp) SCHEDULE(RUNTIME)
  do k = 1, n
    do i = 1, n,b
       do j = k+1,n 
            do ii = i,i+b-1
               temp = A(ii,j,k)
               A(ii,j,k) = A(ii,k,j)
               A(ii,k,j) = temp
            end do
       end do
    end do
  end do
  !$omp end parallel do
  finish = omp_get_wtime()
  
  arraySum = 0.0
  do k = 1, n
    do j = 1, n
      do i = 1, n
        arraySum = arraySum + A(i,j,k)
      end do
    end do
  end do

  print*, "!--------------parallelizing k loop with unrolling i-----------!"
  print *, "Time: ", finish - start, ", Sum: ", arraySum, cnt

!--------------parallelizing i loop------------!
!  cnt = 0
!  start = omp_get_wtime()
!  !$omp parallel do private(k,j,temp) SCHEDULE(RUNTIME)
!  do i = 1, n
!    do k = 1, n
!      do j = k+1, n
!        temp = A(i,j,k)
!        A(i,j,k) = A(i,k,j)
!        A(i,k,j) = temp
!      end do
!    end do
!  end do
!  !$omp end parallel do
!  finish = omp_get_wtime()
  
!  arraySum = 0.0
!  do k = 1, n
!    do j = 1, n
!      do i = 1, n
!        arraySum = arraySum + A(i,j,k)
!      end do
!    end do
!  end do

!  print*, "!--------------parallelizing i loop------------!"
!  print *, "Time: ", finish - start, ", Sum: ", arraySum,cnt
  
!--------------parallelizing k loop------------!
!  b = 1; cnt = 0
!  start = omp_get_wtime()
!  !$omp parallel do private(i,j,ii,temp) SCHEDULE(RUNTIME)
!  do k = 1, n
!    do i = 1, n,b
!       do j = k+1,n 
!            do ii = i,i+b-1
 !              cnt = cnt+1
 !              temp = A(ii,j,k)
 !              A(ii,j,k) = A(ii,k,j)
 !              A(ii,k,j) = temp
 !           end do
 !      end do
 !   end do
 ! end do
 ! !$omp end parallel do
 ! finish = omp_get_wtime()
  
 ! arraySum = 0.0
 ! do k = 1, n
 !   do j = 1, n
 !     do i = 1, n
 !       arraySum = arraySum + A(i,j,k)
 !     end do
 !   end do
 ! end do

 ! print*, "!--------------parallelizing k loop-----------!"
 ! print *, "Time: ", finish - start, ", Sum: ", arraySum, cnt

  deallocate( A )
  stop
end


