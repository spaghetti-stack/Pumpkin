include "globals.mzn";

% Use this editor as a MiniZinc scratch book
% Define the set of allowed values
set of int: valid_values = {1,3,5,7,6,9};

% Declare variables that can only take values from this set
var valid_values: x;
var valid_values: y;
var valid_values: z;
var int: target = 16;  % Replace with your desired sum

% Parameters for global_cardinality_low_up
array[1..6] of int: values = [1,3,5,7,6,9];   % The possible values
array[1..6] of int: lower = [0,0,0,0,0,0];    % Minimum occurrences (can be 0)
array[1..6] of int: upper = [1,1,1,1,1,1];    % Maximum occurrences (1 to be like all_different)

% Add the constraint
constraint x + y + z = target;
%constraint all_different([x, y, z]);
constraint global_cardinality_low_up([x,y,z], values, lower, upper);
%constraint pumpkin_gcc_basic_filter([x,y,z], values, lower, upper);
%constraint pumpkin_gcc_regin([x,y,z], values, lower, upper);



% Additional constraint to ensure 6 and 9 don't both appear
%constraint (sum([1 | v in [x,y,z] where v = 6]) +
%           sum([1 | v in [x,y,z] where v = 9])) <= 1;
constraint count([x,y,z], 6) + count([x,y,z], 9) <= 1;

% Force increasing order to eliminate permutations
constraint increasing([x,y,z]);

% Show all solutions
solve satisfy;

% Output format
output ["x = \(x), y = \(y), z = \(z)\n"];
