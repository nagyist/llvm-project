! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s
program p
   integer, parameter :: n = 256
   real(8) :: a(256)
   type t1
      integer :: x
   end type t1
   !$omp declare mapper(xx : t1 :: nn) map(nn, nn%x)
   !$omp target map(mapper(xx), from:a)
!CHECK: not yet implemented: Support for mapper modifiers is not implemented yet
   do i = 1, n
      a(i) = 4.2
   end do
   !$omp end target
end program p
