program main
  use omp_lib
  implicit none

  real*8 :: start, finish
  integer ( kind = 4 ) i, j, k
  integer seed
  real*8 arraySum, cnt
  real*4 temp
  integer :: n = 5000, m =4
  integer :: jstart, jend, b
  integer :: block, threadID, nthreads
  real*4, DIMENSION(:,:,:), allocatable :: A
                
  allocate( A(m,n,n) )
  seed = 13
  call srand(seed)
  
  print*, "enter m (smaller dimension)"
  read*,m  
  print*, "enter n "
  read*,n  

  do k = 1, n
    do j = 1, n
      do i = 1, m
        A(i,j,k) = mod(irand(), 10)
      end do
    end do
  end do
  
!------------------------serial code-----------------------------------------------!
  start = omp_get_wtime()
  do i = 1, m
        do k = 1,n
           do j = k+1,n
                 temp = A(i,j,k)
                 A(i,j,k) = A(i,k,j)
                 A(i,k,j) = temp
           end do
        end do
  end do
  
  finish = omp_get_wtime()
  
  arraySum = 0.0
  do k = 1, n
    do j = 1, n
      do i = 1, m
        arraySum = arraySum + A(i,j,k)
      end do
    end do
  end do

  print*, "!--------------serial code--------------------------------------------------!"
  print *, "Time: ", finish - start, ", Sum: ", arraySum

print*,"!------------------------------------------------------------------------------!"
print*,"!------------------------------------------------------------------------------!"

!-----------------omp parallel-----------------------------!
  b = 10; cnt = 0
  start = omp_get_wtime()
  
  !$omp parallel private(k,i,nthreads,threadID,block,jstart,jend,temp)
  do k = 1, n-1
     nthreads = omp_get_num_threads()
     threadID = omp_get_thread_num()
     if (threadID == 0) then
        block = (n-k)/nthreads + mod(n-k,nthreads)
        jstart = k+1
     else
        block = (n-k)/nthreads
        jstart = mod(n-k,nthreads) + block * threadID +k+1
     end if
        jend = jstart + block - 1
     do j = jstart,jend 
        do i=1,m 
           temp = A(i,j,k)
           A(i,j,k) = A(i,k,j)
           A(i,k,j) = temp
        end do
     end do
  end do
  !$omp end parallel
  
  finish = omp_get_wtime()
  
  arraySum = 0.0
  do k = 1, n
    do j = 1, n
      do i = 1, m
        arraySum = arraySum + A(i,j,k)
      end do
    end do
  end do

  print*,"!--------------omp parallel----------------------------!"
  print *, "Time: ", finish - start, ", Sum: ", arraySum

!-----------------nested parallel-----------------------------!
!  start = omp_get_wtime()
!  !$omp_set_nested(.true.)  
!  !$omp parallel private(k,i,nthreads,threadID,block,jstart,jend,temp)
!  do k = 1, n-1
!     nthreads = omp_get_num_threads()
!     threadID = omp_get_thread_num()
!     if (threadID == 0) then
!        block = (n-k)/nthreads + mod(n-k,nthreads)
!        jstart = k+1
!     else
!        block = (n-k)/nthreads
!        jstart = mod(n-k,nthreads) + block * threadID +k+1
!     end if
!        jend = jstart + block - 1
!     !$omp do firstprivate(jstart,jend) private(temp,j)
!     do i=1,m 
!        do j = jstart,jend 
!           temp = A(i,j,k)
!           A(i,j,k) = A(i,k,j)
!           A(i,k,j) = temp
!        end do
!     end do
!     !$omp end do
!  end do
!  !$omp end parallel
!  
 ! finish = omp_get_wtime()
  
 ! arraySum = 0.0
 ! do k = 1, n
 !   do j = 1, n
 !     do i = 1, m
 !       arraySum = arraySum + A(i,j,k)
 !     end do
 !   end do
 ! end do

 ! print*,"!--------------nested parallel----------------------------!"
 ! print *, "Time: ", finish - start, ", Sum: ", arraySum

  deallocate( A )
  stop
end


