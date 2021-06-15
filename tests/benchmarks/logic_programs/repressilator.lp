FEATURE p 0 1
FEATURE q 0 1
FEATURE r 0 1
TARGET p 0 1
TARGET q 0 1
TARGET r 0 1

p(0,T) :- .
p(1,T) :- q(1,T-1).
q(0,T) :- p(0,T-1).
q(0,T) :- r(0,T-1).
q(1,T) :- p(1,T-1), r(1,T-1).
r(0,T) :- p(1,T-1).
r(1,T) :- p(0,T-1).
