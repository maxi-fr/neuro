function [gamma, metadata] = generate_roast_gamma(montages, varargin)
% GENERATE_ROAST_GAMMA Computes the spatial projection matrix (gamma) for electrode
% configurations using the ROAST library and equation (8) from Yu et al. (2024):
%   "Closed-loop transcranial electrical stimulation for inhibiting epileptic activity
%    propagation: a whole-brain model study", Nonlinear Dynamics.
%
% Equation (8):
%   gamma_i = E(x_i, y_i, z_i) * sign(V(x_i, y_i, z_i) - V_med)
% where E is electric field magnitude, V is voltage, and V_med = (V_max + V_min)/2.
%
% Usage:
%   [gamma, metadata] = generate_roast_gamma({'CP5', -1, 'Ex8', 1})
%   [gamma, metadata] = generate_roast_gamma({{'CP5', -1, 'Ex8', 1}, {'PO3', -1, 'Ex8', 1}}, ...
%                           'elecType', 'pad', 'elecSize', [50 30 3], 'outputFile', 'data/roast_gamma.mat')
%
% Inputs:
%   montages     - Single electrode pair (e.g. {'CP5', -1, 'Ex8', 1} or {'CP5', 'Ex8'})
%                  or a cell array of pairs for multi-electrode configurations.
%
% Name-Value Parameters:
%   'subj'         - MRI subject for ROAST (default: 'example/MNI152_T1_1mm.nii')
%   'capType'      - Cap system: 'biosemi', '1005', '1010' (auto-detected if empty)
%   'elecType'     - Electrode shape: 'pad' (default), 'disc', or 'ring'
%   'elecSize'     - Electrode dimensions: [50 30 3] (default for pad in mm)
%   'mniCoords'    - N_regions x 3 matrix of region MNI RAS coordinates (mm).
%   'regionLabels' - Cell array of region labels (length N_regions).
%   'outputFile'   - Output file path for .mat saving (default: 'data/roast_gamma.mat')
%
% Outputs:
%   gamma        - Matrix of shape (n_montages, n_regions) with voltage perturbation factors
%   metadata     - Struct containing montages, region labels, coordinates, and options

    % 1. Add ROAST to MATLAB path
    matlabDir = fileparts(mfilename('fullpath'));
    roastDir = fullfile(matlabDir, 'roast-4.0');
    if exist(roastDir, 'dir')
        addpath(genpath(roastDir));
    else
        error('ROAST directory not found at: %s', roastDir);
    end

    % 2. Parse input arguments
    p = inputParser;
    addRequired(p, 'montages');
    addParameter(p, 'subj', 'example/MNI152_T1_1mm.nii', @ischar);
    addParameter(p, 'capType', '', @ischar);
    addParameter(p, 'elecType', 'pad', @(x) ischar(x) || iscell(x));
    addParameter(p, 'elecSize', [50 30 3], @isnumeric);
    addParameter(p, 'mniCoords', [], @isnumeric);
    addParameter(p, 'regionLabels', {}, @iscell);
    addParameter(p, 'outputFile', 'data/roast_gamma.mat', @ischar);
    parse(p, montages, varargin{:});

    subj = p.Results.subj;
    % Resolve subject to full absolute path in roastDir
    if exist(fullfile(roastDir, subj), 'file')
        subj = fullfile(roastDir, subj);
    elseif exist(fullfile(roastDir, 'example', subj), 'file')
        subj = fullfile(roastDir, 'example', subj);
    elseif ~isempty(which(subj))
        subj = which(subj);
    end
    [subjDir, subjFile, subjExt] = fileparts(subj);
    if isempty(subjDir) || strcmp(subjDir, '.') || strcmp(subjDir, 'example')
        subjDir = fullfile(roastDir, 'example');
        subj = fullfile(subjDir, [subjFile subjExt]);
    end

    userCapType = p.Results.capType;
    elecType = p.Results.elecType;
    elecSize = p.Results.elecSize;
    mniCoords = p.Results.mniCoords;
    regionLabels = p.Results.regionLabels;
    outputFile = p.Results.outputFile;

    % Standardize montages to a cell array of recipe cell arrays
    rawMontages = p.Results.montages;
    if isempty(rawMontages)
        error('Montages input cannot be empty.');
    end
    if iscell(rawMontages) && ~isempty(rawMontages) && ischar(rawMontages{1})
        % Single recipe passed: e.g. {'CP5', -1, 'Ex8', 1} or {'CP5', 'Ex8'}
        montageList = {normalize_recipe(rawMontages)};
    elseif iscell(rawMontages) && iscell(rawMontages{1})
        % List of recipes passed
        montageList = cell(size(rawMontages));
        for k = 1:length(rawMontages)
            montageList{k} = normalize_recipe(rawMontages{k});
        end
    else
        error('Invalid montages format. Provide a cell array of electrode pairs.');
    end

    % Load region coordinates if not explicitly provided
    if isempty(mniCoords)
        [mniCoords, regionLabels] = load_default_region_coords(matlabDir);
    end
    nRegions = size(mniCoords, 1);

    nMontages = length(montageList);
    gamma = zeros(nMontages, nRegions);

    % 3. Process each montage configuration through ROAST
    for k = 1:nMontages
        recipe = montageList{k};

        % Default capType to 1005 (which contains standard scalp and Ex1..Ex8 electrodes)
        capType = userCapType;
        if isempty(capType)
            capType = '1005';
        end

        % Map aliases if needed (e.g. TP9 -> TP7 for 1005 cap, EX_NECK -> Ex8 / nk4)
        recipe = sanitize_electrode_names(recipe, capType);

        fprintf('\n>>> Processing montage %d/%d (capType: %s): %s <<<\n', ...
                k, nMontages, capType, recipe_to_str(recipe));

        % Run ROAST simulation inside roastDir with full absolute subject path
        origDir = pwd;
        cd(roastDir);
        cleanup = onCleanup(@() cd(origDir));

        roast(subj, recipe, 'capType', capType, 'elecType', elecType, 'elecSize', elecSize);

        % Resolve output file names generated by ROAST
        [subjDir, subjName] = fileparts(subj);
        if isempty(subjDir), subjDir = pwd; end

        % ROAST saves result files using a tag derived from recipe and options
        % Find the created roastResult.mat file in subjDir or current folder
        resultFiles = dir(fullfile(subjDir, [subjName '_*_roastResult.mat']));
        if isempty(resultFiles)
            resultFiles = dir(fullfile(pwd, [subjName '_*_roastResult.mat']));
        end

        if isempty(resultFiles)
            error('Could not locate ROAST result file (*_roastResult.mat) after simulation.');
        end

        % Sort by date to pick the most recent result file
        [~, latestIdx] = max([resultFiles.datenum]);
        resultFile = fullfile(resultFiles(latestIdx).folder, resultFiles(latestIdx).name);

        fprintf('Loading ROAST results from: %s\n', resultFile);
        resData = load(resultFile, 'vol_all', 'ef_mag');
        vol_all = resData.vol_all;
        ef_mag = resData.ef_mag;

        % Load NIfTI affine matrix to map MNI RAS coordinates to voxel space
        niiFile = strrep(resultFile, '_roastResult.mat', '_v.nii');
        if ~exist(niiFile, 'file')
            niiFile = fullfile(subjDir, [subjName '_v.nii']);
        end

        if exist(niiFile, 'file') && exist('load_untouch_nii', 'file')
            nii = load_untouch_nii(niiFile);
            affine = [nii.hdr.hist.srow_x; ...
                      nii.hdr.hist.srow_y; ...
                      nii.hdr.hist.srow_z; ...
                      0 0 0 1];
        else
            % Default 1mm MNI affine if NIfTI header not accessible
            affine = eye(4);
        end

        % Compute V_med
        vMax = max(vol_all(:));
        vMin = min(vol_all(:));
        vMed = (vMax + vMin) / 2.0;

        % Map MNI RAS coordinates (x,y,z) to voxel grid indices (i,j,k)
        % Voxel = inv(affine) * [MNI; 1]
        invAffine = inv(affine);
        mniHom = [mniCoords, ones(nRegions, 1)]';
        voxHom = invAffine * mniHom;
        voxCoords = voxHom(1:3, :)';

        % Grid definitions for 3D interpolation
        [dim1, dim2, dim3] = size(vol_all);
        [X, Y, Z] = ndgrid(1:dim1, 1:dim2, 1:dim3);

        % Interpolate electric field magnitude E and potential V at region locations
        E_regions = interp3(Y, X, Z, ef_mag, voxCoords(:,2), voxCoords(:,1), voxCoords(:,3), 'linear', 0);
        V_regions = interp3(Y, X, Z, vol_all, voxCoords(:,2), voxCoords(:,1), voxCoords(:,3), 'linear', 0);

        % Calculate gamma according to Yu et al. (2024) Eq. 8
        % gamma_i = E(x_i, y_i, z_i) * sign(V(x_i, y_i, z_i) - V_med)
        polarity = sign(V_regions - vMed);
        % Ensure polarity is non-zero (default to +1 for exact V_med boundary)
        polarity(polarity == 0) = 1;

        gamma(k, :) = E_regions .* polarity;
    end

    % Build metadata structure
    metadata = struct();
    metadata.montages = montageList;
    metadata.regionLabels = regionLabels;
    metadata.mniCoords = mniCoords;
    metadata.elecType = elecType;
    metadata.elecSize = elecSize;
    metadata.subj = subj;

    % Save output to MAT file if output file specified
    if ~isempty(outputFile)
        outDir = fileparts(outputFile);
        if ~isempty(outDir) && ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        save(outputFile, 'gamma', 'metadata', '-v7.3');
        fprintf('\nSuccessfully saved ROAST gamma matrix (%d x %d) to %s\n', ...
                nMontages, nRegions, outputFile);
    end
end

%% Helper: Normalize recipe format
function recipe = normalize_recipe(raw)
    if length(raw) == 2 && ischar(raw{1}) && ischar(raw{2})
        % e.g. {'CP5', 'Ex8'} -> {'CP5', -1, 'Ex8', 1}
        recipe = {raw{1}, -1, raw{2}, 1};
    else
        recipe = raw;
    end
end

%% Helper: Convert recipe to readable string
function str = recipe_to_str(recipe)
    strParts = {};
    for i = 1:2:length(recipe)
        strParts{end+1} = sprintf('%s (%.1fmA)', recipe{i}, recipe{i+1}); %#ok<AGROW>
    end
    str = strjoin(strParts, ', ');
end

%% Helper: Sanitize electrode names to match capType
function recipe = sanitize_electrode_names(recipe, capType) %#ok<INUSD>
    for i = 1:2:length(recipe)
        name = recipe{i};
        if strcmpi(name, 'TP9')
            recipe{i} = 'TP7'; % TP9 maps to TP7 in ROAST 10-05 cap
        elseif strcmpi(name, 'EX_NECK')
            recipe{i} = 'Ex8'; % EX_NECK maps to Ex8 in ROAST 10-05 cap
        end
    end
end

%% Helper: Load default region coordinates from TVB export geometry
function [coords, labels] = load_default_region_coords(matlabDir)
    repoDir = fileparts(matlabDir);
    matFile = fullfile(repoDir, 'data', 'tvb_geometry.mat');

    if exist(matFile, 'file')
        data = load(matFile);
        coords = data.centres_mni_ras;
        labels = data.region_labels;
    else
        error(['Region coordinates not provided and %s not found. ', ...
               'Please pass mniCoords or run scripts/export_tvb_geometry.py.'], matFile);
    end
end
