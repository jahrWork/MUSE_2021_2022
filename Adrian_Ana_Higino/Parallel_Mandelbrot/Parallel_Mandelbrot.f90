!  Parallel_Mandelbrot.f90 
!
!  FUNCTIONS:
!  Parallel_Mandelbrot - Entry point of console application.
!

!****************************************************************************
!
!  PROGRAM: Parallel_Mandelbrot
!
!  PURPOSE:  Entry point for the console application.
!
!****************************************************************************

program Parallel_Mandelbrot


use mandelbrot_function


implicit none

    ! Variables
    integer, parameter:: n= 800, n_iter= 40 ! 800, 40 ! n: number of squares to be painted
    integer:: k(0:n ,0:n)
    real(8):: start, finish, time


    ! Body of Mandelbrot_Set 

    ! Parallel Mandelbrot
        k = 0
        write(*,*) "Parallel code"
        call mandelbrot_f_Parallel(n, n_iter, k)

 
    call sleep(5)
    ! Non-Parallel Mandelbrot
        k = 0
        write(*,*) ""
        write(*,*) "Non-parallel code"
        call mandelbrot_f(n, n_iter, k) ! for its time, execute it alone


end program Parallel_Mandelbrot


