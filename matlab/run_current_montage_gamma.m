% Script to generate the gamma matrix for the current active project montage:
%   Target Electrodes: [TP9, CP5, EX_NECK]
%   Cathodes: TP9, CP5 (-1 mA each)
%   Return: Ex8 (+1 mA extracephalic neck return in ROAST)
%
% Uses ROAST 4.0 and equation (8) from Yu et al. (2024):
%   gamma_i = E(x_i, y_i, z_i) * sign(V(x_i, y_i, z_i) - V_med)

% 1. Define active montage electrode pairs (10-05 cap: TP7, CP5, Ex8)
montages = { ...
    {'TP7', -1, 'Ex8', 1}, ...  % Channel 1: TP7 (TP9 equivalent) vs Ex8 neck return
    {'CP5', -1, 'Ex8', 1}   ...  % Channel 2: CP5 vs Ex8 neck return
};

% 2. Specify output path
outputFile = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'roast_gamma.mat');

fprintf('=======================================================\n');
fprintf('  Generating ROAST Gamma Matrix for Active Montage     \n');
fprintf('  Montage Pairs:\n');
fprintf('    1) TP9 (-1 mA) <-> Ex8 (+1 mA)\n');
fprintf('    2) CP5 (-1 mA) <-> Ex8 (+1 mA)\n');
fprintf('=======================================================\n\n');

% 3. Run generator
[gamma, metadata] = generate_roast_gamma(montages, ...
    'elecType', 'pad', ...
    'elecSize', [50 30 3], ...
    'outputFile', outputFile);

fprintf('\n>>> SUCCESS: Generated gamma matrix shape (%d x %d) saved to %s <<<\n', ...
        size(gamma, 1), size(gamma, 2), outputFile);
