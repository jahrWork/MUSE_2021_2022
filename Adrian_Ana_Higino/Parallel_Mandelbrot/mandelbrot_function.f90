module mandelbrot_function

USE OMP_LIB

implicit none 

private 
public ::  mandelbrot_f_Parallel, mandelbrot_f
contains 

subroutine mandelbrot_f_Parallel(n, n_iter, k_out)
    ! in, out, inout variables
    integer, intent(in) :: n, n_iter
    integer, intent(out) :: k_out(1:n, 1:n)

    ! non parallel variables
    integer :: i, j, ii, it
    real(8), parameter :: x0= -2, x1= 1, y0= -1.5, y1= 1.5
    complex(8) :: z(1:n, 1:n), z_abs(1:n, 1:n)
    real(8) :: x(1:n), y(1:n), t0, tf

    ! Time counter
    integer :: rate
    real(8) :: tCPU0, tCPUf
    integer :: treal0, trealf
    
    ! parallel variables
    INTEGER, PARAMETER :: number_of_threads = 4
    INTEGER :: thread_id, k(1:n, 1:n)
    !REAL(8) :: 
    COMPLEX(8) :: c(1:n, 1:n)

    
    CALL system_clock(count_rate=rate)


    call CPU_TIME(tCPU0)
    call SYSTEM_CLOCK(treal0)

    call OMP_SET_NUM_THREADS(number_of_threads) ! threads go from 0 to number-1

    c= cmplx(0,0); z= cmplx(0,0); k_out= 0; k= 0 ! create them
   
    ! observed range
    x= [( x0 + (x1-x0)/(n-1) * (it-1), it= 1,n )]
        ! x avances through the rows
    y= [( y0 + (y1-y0)/(n-1) * (it-1), it= 1,n )]
        ! y advances through the collumns
    
    ! create the complex region to analize
    do i = 1, n

        !$OMP PARALLEL PRIVATE(thread_id) SHARED(c)
            thread_id = OMP_GET_THREAD_NUM()
            !$OMP DO
            DO j = 1, n
                c(i,j) = cmplx(x(i), y(j))
            END DO
            !$OMP END DO

        !$OMP END PARALLEL

    enddo

    ! calculate the Mandelbrot matrix to be painted
    do ii = 1, n_iter
        z = z**2 + c
        do i = 1, n
            !$OMP PARALLEL PRIVATE(thread_id) SHARED(k)
                !$OMP DO
                DO j = 1, n

                    if (abs(z(i, j))>2.0 .and. k(i, j)==0) k(i, j) = n_iter - ii

                END DO
                !$OMP END DO

            !$OMP END PARALLEL

        enddo 
    enddo
    
    k_out = k;    

    call CPU_TIME(tCPUf)
    call SYSTEM_CLOCK(trealf)

    write(*,*) "Number of threads: ", number_of_threads
    write(*,*) "CPU time: ", tCPUf - tCPU0 
    write(*,*) "Real time: ", real(trealf - treal0)/real(rate)


endsubroutine


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine mandelbrot_f(n, n_iter, k)
    integer, intent(in):: n, n_iter
    integer, intent(out):: k(1:n, 1:n)

    integer:: i, j, ii, it
    real(8), parameter:: x0= -2, x1= 1, y0= -1.5, y1= 1.5
    complex(8):: c(1:n, 1:n), z(1:n, 1:n), z_abs(1:n, 1:n)
    real(8):: x(1:n), y(1:n)

    ! Time counter
    integer :: rate
    real(8) :: tCPU0, tCPUf
    integer :: treal0, trealf

    
    CALL system_clock(count_rate=rate)


    call CPU_TIME(tCPU0)
    call SYSTEM_CLOCK(treal0)
    
    c= cmplx(0,0); z= cmplx(0,0); k= 0 !initialization
   
    x= [( x0 + (x1-x0)/(n-1) * (it-1), it= 1,n )]
    y= [( y0 + (y1-y0)/(n-1) * (it-1), it= 1,n )]
    
    do i = 1, n
        do j = 1, n

            c(i, j) = cmplx(x(i), y(j))

        enddo
    enddo

    
    ! calculate the Mandelbrot matrix to be painted
    do ii = 1, n_iter
        z = z**2 + c
        do i = 1, n

                do j = 1, n
                    if (abs(z(i, j))>2.0 .and. k(i, j)==0) k(i, j) = n_iter - ii
                end do

        enddo 
    enddo
    

    call CPU_TIME(tCPUf)
    call SYSTEM_CLOCK(trealf)

    write(*,*) "CPU time: ", tCPUf - tCPU0 
    write(*,*) "Real time: ", real(trealf - treal0)/real(rate)


endsubroutine


endmodule