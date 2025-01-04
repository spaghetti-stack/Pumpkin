%predicate fzn_global_cardinality_low_up(array[int] of var int: x,
%                                        array[int] of int: cover,
%                                        array[int] of int: lbound,
%                                        array[int] of int: ubound)
%  = pumpkin_gcc_regin(x, cover, lbound, ubound);

%predicate pumpkin_global_cardinality_low_up(array[int] of var int: x,
%                                        array[int] of int: cover,
%                                        array[int] of int: lbound,
%                                        array[int] of int: ubound);

predicate pumpkin_gcc_bruteforce(array[int] of var int: x,
                                        array[int] of int: cover,
                                        array[int] of int: lbound,
                                        array[int] of int: ubound);

predicate pumpkin_gcc_basic_filter(array[int] of var int: x,
                                        array[int] of int: cover,
                                        array[int] of int: lbound,
                                        array[int] of int: ubound);
                                        
predicate pumpkin_gcc_regin(array[int] of var int: x,
                                        array[int] of int: cover,
                                        array[int] of int: lbound,
                                        array[int] of int: ubound);