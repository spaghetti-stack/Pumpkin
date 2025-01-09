include "globals.mzn";

% Define the array of variables
array[1..5] of var 1..3: x;

% Define the values we want to count
array[1..3] of int: values = [1, 2, 3];

% Define minimum occurrences for each value
array[1..3] of int: low = [1, 1, 1];  % each value must appear at least once

% Define maximum occurrences for each value
array[1..3] of int: up = [2, 2, 2];   % each value can appear at most twice

%include "pumpkin_gcc.mzn";

constraint global_cardinality_low_up(x, values, low, up);

solve satisfy;

output ["x = ", show(x)];