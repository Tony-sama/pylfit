FEATURE p2 0 1
FEATURE q2 0 1
FEATURE r2 0 1
FEATURE p1 0 1
FEATURE q1 0 1
FEATURE r1 0 1
TARGET p 0 1
TARGET q 0 1
TARGET r 0 1

p(0,T) :- q1(0,T-1).
p(1,T) :- q1(1,T-2).
q(0,T) :- p2(0,T-1).
q(0,T) :- r1(0,T-1).
q(1,T) :- p2(1,T-1), r1(1,T-1).
r(0,T) :- p2(1,T-2).
r(1,T) :- p2(0,T-2).
