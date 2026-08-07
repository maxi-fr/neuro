function [leadfield_3d, metadata] = generate_roast_leadfield_3d(varargin)
% GENERATE_ROAST_LEADFIELD_3D Computes the 3D electric field leadfield matrix (L)
% for 63 input channels (62 scalp electrodes + Ex8 return) to
% 76 output Jansen-Rit node positions using the ROAST library.
%
% Outputs:
%   leadfield_3d - Array of shape (63, 76, 3) containing (Ex, Ey, Ez) at each JR node
%                  per +1 mA stimulation channel relative to Ex8 reference return.
%                  Row 63 (Ex8 anode) is defined as zeros (reference ground).
%   metadata     - Struct containing channel labels, region labels, coordinates, and options.
%                  channelLabels holds the requested (TVB) names; roastLabels holds the
%                  ROAST cap electrode actually simulated for each row, and labelDots the
%                  cosine between the two positions.
%
% Usage:
%   [L, meta] = generate_roast_leadfield_3d('outputFile', 'data/roast_leadfield_3d.mat');

    % 1. Add ROAST to MATLAB path
    matlabDir = fileparts(mfilename('fullpath'));
    roastDir = fullfile(matlabDir, 'roast-4.0');
    if exist(roastDir, 'dir')
        addpath(genpath(roastDir));
    else
        error('ROAST directory not found at: %s', roastDir);
    end

    % 2. Parse input parameters
    p = inputParser;
    addParameter(p, 'subj', 'example/MNI152_T1_1mm.nii', @ischar);
    addParameter(p, 'capType', '1005', @ischar);
    addParameter(p, 'elecType', 'pad', @(x) ischar(x) || iscell(x));
    addParameter(p, 'elecSize', [50 30 3], @isnumeric);
    addParameter(p, 'returnElectrode', 'Ex8', @ischar);
    % Ex8 sits below the inion; without zeropadding it falls outside the MRI and ROAST
    % silently returns a field solved on a truncated head (see ROAST README, Example 22).
    addParameter(p, 'zeroPadding', 60, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'minLabelDot', 0.98, @isnumeric);
    addParameter(p, 'mniCoords', [], @isnumeric);
    addParameter(p, 'regionLabels', {}, @iscell);
    addParameter(p, 'channelLabels', {}, @iscell);
    addParameter(p, 'outputFile', 'data/roast_leadfield_3d.mat', @ischar);
    parse(p, varargin{:});

    subj = p.Results.subj;
    if exist(fullfile(roastDir, subj), 'file')
        subj = fullfile(roastDir, subj);
    elseif exist(fullfile(roastDir, 'example', subj), 'file')
        subj = fullfile(roastDir, 'example', subj);
    elseif ~isempty(which(subj))
        subj = which(subj);
    end

    capType = p.Results.capType;
    elecType = p.Results.elecType;
    elecSize = p.Results.elecSize;
    returnElec = p.Results.returnElectrode;
    zeroPadding = p.Results.zeroPadding;
    minLabelDot = p.Results.minLabelDot;
    mniCoords = p.Results.mniCoords;
    regionLabels = p.Results.regionLabels;
    scalpLabels = p.Results.channelLabels;
    outputFile = p.Results.outputFile;
    if ~isempty(outputFile) && ~java.io.File(outputFile).isAbsolute()
        outputFile = fullfile(pwd, outputFile);
    end

    % Load region coordinates & default channel labels if not provided
    [mniCoords, regionLabels, defaultScalpLabels, regionNormals, channelPositions] = ...
        load_geometry_data(matlabDir, mniCoords, regionLabels, scalpLabels);
    nRegions = size(mniCoords, 1);

    if isempty(scalpLabels)
        scalpLabels = defaultScalpLabels;
    end
    scalpLabels(strcmpi(scalpLabels, returnElec)) = [];
    nScalp = length(scalpLabels);

    % All channels: scalp channels + 1 return electrode
    allChannelLabels = [scalpLabels(:); {returnElec}];
    nChannels = length(allChannelLabels);

    % Resolve every channel against the ROAST cap BEFORE the (multi-hour) loop, so an
    % unrecognised label fails now instead of on iteration 50.
    [roastLabels, labelDots] = resolve_electrodes( ...
        allChannelLabels, channelPositions, returnElec, roastDir, capType, minLabelDot);

    fprintf('\n=======================================================\n');
    fprintf(' Generating ROAST 3D Leadfield Matrix (%d channels x %d regions x 3)\n', nChannels, nRegions);
    fprintf(' Return electrode: %s | zeropadding: %d\n', returnElec, zeroPadding);
    substituted = find(~strcmpi(allChannelLabels(:), roastLabels(:)));
    for s = substituted(:)'
        fprintf(' Substituted %s -> %s (cos = %.4f)\n', allChannelLabels{s}, roastLabels{s}, labelDots(s));
    end
    fprintf('=======================================================\n\n');

    leadfield_3d = zeros(nChannels, nRegions, 3);

    % Enter the ROAST directory once. Re-creating the onCleanup inside the loop would destroy
    % the previous one and cd back out, leaving every iteration after the first in the wrong cwd.
    origDir = pwd;
    cd(roastDir);
    cleanup = onCleanup(@() cd(origDir)); %#ok<NASGU>

    % Run ROAST for each of the 62 scalp electrodes (+1 mA) with the return at -1 mA
    for k = 1:nScalp
        scalpElec = roastLabels{k};
        recipe = {scalpElec, 1, roastLabels{nChannels}, -1};

        fprintf('\n>>> [%d/%d] Simulating basis montage: %s (+1mA) vs %s (-1mA) <<<\n', ...
                k, nScalp, scalpElec, roastLabels{nChannels});

        roast(subj, recipe, 'capType', capType, 'elecType', elecType, 'elecSize', elecSize, ...
              'zeropadding', zeroPadding);

        [subjDir, subjName] = fileparts(subj);
        if isempty(subjDir), subjDir = pwd; end

        resultFiles = dir(fullfile(subjDir, [subjName '_*_roastResult.mat']));
        if isempty(resultFiles)
            resultFiles = dir(fullfile(pwd, [subjName '_*_roastResult.mat']));
        end

        if isempty(resultFiles)
            error('Could not locate ROAST result file after simulation of %s.', scalpElec);
        end

        [~, latestIdx] = max([resultFiles.datenum]);
        resultFile = fullfile(resultFiles(latestIdx).folder, resultFiles(latestIdx).name);

        resData = load(resultFile);
        if ~isfield(resData, 'ef_all')
            error(['ROAST result %s has no ef_all field. Deriving E from gradient(-vol_all) ' ...
                   'would be in V/voxel, not V/m.'], resultFile);
        end

        affine = read_nifti_affine(resultFile, subjDir, subjName);

        % Compute voxel coordinates for region centres
        invAffine = inv(affine);
        mniHom = [mniCoords, ones(nRegions, 1)]';
        voxHom = invAffine * mniHom;
        voxCoords = voxHom(1:3, :)';

        ef_all = resData.ef_all; % (dim1, dim2, dim3, 3)
        [dim1, dim2, dim3, ~] = size(ef_all);
        [X, Y, Z] = ndgrid(1:dim1, 1:dim2, 1:dim3);

        Ex_grid = ef_all(:,:,:,1);
        Ey_grid = ef_all(:,:,:,2);
        Ez_grid = ef_all(:,:,:,3);

        if any(voxCoords(:,1) < 1 | voxCoords(:,1) > dim1 | ...
               voxCoords(:,2) < 1 | voxCoords(:,2) > dim2 | ...
               voxCoords(:,3) < 1 | voxCoords(:,3) > dim3)
            error(['Region centres fall outside the ROAST volume for %s. The MNI-to-voxel ' ...
                   'affine and the geometry export disagree.'], scalpElec);
        end

        Ex_regions = interp3(Y, X, Z, Ex_grid, voxCoords(:,2), voxCoords(:,1), voxCoords(:,3), 'linear', 0);
        Ey_regions = interp3(Y, X, Z, Ey_grid, voxCoords(:,2), voxCoords(:,1), voxCoords(:,3), 'linear', 0);
        Ez_regions = interp3(Y, X, Z, Ez_grid, voxCoords(:,2), voxCoords(:,1), voxCoords(:,3), 'linear', 0);

        leadfield_3d(k, :, 1) = Ex_regions;
        leadfield_3d(k, :, 2) = Ey_regions;
        leadfield_3d(k, :, 3) = Ez_regions;
    end

    % Row 63 (return electrode) is the zero reference
    leadfield_3d(nChannels, :, :) = 0;

    % Build metadata
    metadata = struct();
    metadata.channelLabels = allChannelLabels;
    metadata.roastLabels = roastLabels;
    metadata.labelDots = labelDots;
    metadata.returnElectrode = returnElec;
    metadata.regionLabels = regionLabels;
    metadata.mniCoords = mniCoords;
    metadata.regionNormals = regionNormals;
    metadata.normalsFrame = 'mni_ras';
    metadata.elecType = elecType;
    metadata.elecSize = elecSize;
    metadata.zeroPadding = zeroPadding;
    metadata.subj = subj;

    if ~isempty(outputFile)
        outDir = fileparts(outputFile);
        if ~isempty(outDir) && ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        save(outputFile, 'leadfield_3d', 'metadata', '-v7.3');
        fprintf('\nSaved ROAST 3D Leadfield matrix (%d x %d x 3) to %s\n', ...
                nChannels, nRegions, outputFile);
    end
end

%% Helper: Resolve requested electrode labels to ROAST cap electrodes
function [roastLabels, dots] = resolve_electrodes(requested, positions, returnElec, roastDir, capType, minDot)
    switch lower(capType)
        case {'1020','1010','1005'}, sheet = '10-05';
        case 'biosemi',              sheet = 'BioSemi';
        case 'egi',                  sheet = 'EGI';
        otherwise, error('Unsupported capType: %s', capType);
    end

    capFile = fullfile(roastDir, 'capInfo.xlsx');
    cap = table2cell(readtable(capFile, 'Sheet', sheet, 'ReadVariableNames', false));
    % ROAST reads capInfo with default options, so row 1 becomes a header and is not
    % selectable as an electrode. Drop it here to match the pool ROAST accepts.
    capNames = cellfun(@char, cap(2:end, 1), 'UniformOutput', false);
    capXYZ = cell2mat(cap(2:end, 2:4));
    capUnit = capXYZ ./ vecnorm(capXYZ, 2, 2);

    nReq = numel(requested);
    if size(positions, 1) ~= nReq - 1
        error('Geometry export has %d channel positions but %d scalp channels were requested.', ...
              size(positions, 1), nReq - 1);
    end

    roastLabels = cell(nReq, 1);
    dots = zeros(nReq, 1);

    for i = 1:nReq
        name = requested{i};
        exact = find(strcmpi(capNames, name), 1);
        if ~isempty(exact)
            roastLabels{i} = capNames{exact};
            dots(i) = 1;
            continue;
        end
        if strcmpi(name, returnElec)
            error('Return electrode %s is not in the ROAST %s cap.', returnElec, sheet);
        end
        % Nearest cap electrode by direction from the head centre.
        u = positions(i, :) / norm(positions(i, :));
        allDots = capUnit * u';
        [bestDot, bestIdx] = max(allDots);
        if bestDot < minDot
            error(['No ROAST %s electrode within cos >= %.3f of %s (best: %s at %.4f). ' ...
                   'Pick a substitute explicitly.'], sheet, minDot, name, capNames{bestIdx}, bestDot);
        end
        roastLabels{i} = capNames{bestIdx};
        dots(i) = bestDot;
    end

    [uniqueLabels, ~, groupIdx] = unique(lower(roastLabels));
    counts = accumarray(groupIdx, 1);
    if any(counts > 1)
        dup = uniqueLabels{find(counts > 1, 1)};
        error('Electrode %s was selected for more than one channel; leadfield rows would be identical.', dup);
    end
end

%% Helper: Read the voxel-to-MNI affine from the ROAST output NIfTI
function affine = read_nifti_affine(resultFile, subjDir, subjName)
    if ~exist('load_untouch_nii', 'file')
        error('load_untouch_nii not found on the path; ROAST''s NIfTI toolbox is required.');
    end

    niiFile = strrep(resultFile, '_roastResult.mat', '_v.nii');
    if ~exist(niiFile, 'file')
        niiFile = fullfile(subjDir, [subjName '_v.nii']);
    end
    if ~exist(niiFile, 'file')
        error('Could not locate the ROAST voltage NIfTI needed for the voxel affine (%s).', niiFile);
    end

    nii = load_untouch_nii(niiFile);
    affine = [nii.hdr.hist.srow_x; ...
              nii.hdr.hist.srow_y; ...
              nii.hdr.hist.srow_z; ...
              0 0 0 1];
    rotation = affine(1:3, 1:3);
    if ~any(rotation(:))
        error('NIfTI %s has an empty sform; voxel coordinates cannot be derived.', niiFile);
    end
end

%% Helper: Load default geometry data
function [coords, labels, scalpLabels, regionNormals, channelPositions] = ...
        load_geometry_data(matlabDir, userCoords, userLabels, userScalp)
    repoDir = fileparts(matlabDir);
    matFile = fullfile(repoDir, 'data', 'tvb_geometry.mat');

    coords = userCoords;
    labels = userLabels;
    scalpLabels = userScalp;

    if ~exist(matFile, 'file')
        error('Geometry data missing: %s not found. Run scripts/export_tvb_geometry.py.', matFile);
    end

    data = load(matFile);
    if isempty(coords), coords = data.centres_mni_ras; end
    if isempty(labels), labels = cellstr(data.region_labels); end
    defaultScalp = cellstr(data.channel_labels);
    defaultPositions = data.channel_positions_mni_ras;

    if isempty(scalpLabels)
        scalpLabels = defaultScalp;
        channelPositions = defaultPositions;
    else
        % Filter positions to match requested scalp labels
        channelPositions = zeros(length(scalpLabels), 3);
        for i = 1:length(scalpLabels)
            idx = find(strcmpi(defaultScalp, scalpLabels{i}), 1);
            if ~isempty(idx)
                channelPositions(i, :) = defaultPositions(idx, :);
            else
                channelPositions(i, :) = [NaN, NaN, NaN];
            end
        end
    end
    regionNormals = data.region_normals_mni_ras;
end
