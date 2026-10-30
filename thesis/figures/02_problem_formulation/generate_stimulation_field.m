% Generate the PF-05 ROAST example; remove thesis/tmp/roast_pf05_generation after export.
figureDir = fileparts(mfilename('fullpath'));
thesisDir = fileparts(fileparts(figureDir));
roastRoot = fullfile(fileparts(thesisDir), 'matlab', 'roast-4.0');
scratch = fullfile(thesisDir, 'tmp', 'roast_pf05_generation');
if ~isfolder(scratch), mkdir(scratch); end
addpath(genpath(roastRoot));
set(groot, 'DefaultFigureVisible', 'off');
ISO2MESH_TEMP = scratch;
ISO2MESH_SESSION = 'pf05_';
sourceNames = {'MNI152_T1_1mm.nii', 'MNI152_T1_1mm_ras.nii', ...
    'MNI152_T1_1mm_ras_padded20.nii', ...
    'MNI152_T1_1mm_ras_padded20_T1orT2_seg8.mat', ...
    'MNI152_T1_1mm_ras_padded20_T1orT2_SPM_masks.nii', ...
    'MNI152_T1_1mm_ras_padded20_T1orT2_SPM_masks_MNI.nii', ...
    'MNI152_T1_1mm_ras_padded20_T1orT2_SPM_MNI.nii'};
for k = 1:numel(sourceNames)
    name = sourceNames{k};
    copyfile(fullfile(roastRoot, 'example', name), fullfile(scratch, name));
end
previousDir = pwd;
restoreDir = onCleanup(@() cd(previousDir));
cd(roastRoot);
roast(fullfile(scratch, 'MNI152_T1_1mm.nii'), {'TPP9', 1, 'Ex8', -1}, ...
    'capType', '1005', 'elecType', 'pad', 'elecSize', [50 30 3], ...
    'elecOri', 'lr', 'zeropadding', 20, 'simulationTag', 'pf05_tpp9_ex8');
cd(previousDir);

% Read the solved nodal field directly: mV/mm equals V/m.
base = fullfile(scratch, 'MNI152_T1_1mm_pf05_tpp9_ex8');
mesh = load([base '.mat'], 'node', 'face');
seg = load(fullfile(scratch, 'MNI152_T1_1mm_ras_padded20_T1orT2_seg8.mat'), 'image');
vertices = mesh.node(:, 1:3);
for k = 1:3
    vertices(:, k) = vertices(:, k) / seg.image(1).mat(k, k);
end
vertices = (seg.image(1).mat * [vertices, ones(size(vertices, 1), 1)]')';
vertices = vertices(:, 1:3);
fid = fopen([base '_e.pos']);
fgetl(fid);
field = textscan(fid, '%d %f %f %f');
fclose(fid);
magnitude = nan(size(vertices, 1), 1);
magnitude(field{1}) = sqrt(field{2}.^2 + field{3}.^2 + field{4}.^2);
cortex = mesh.face(mesh.face(:, 4) == 2, 1:3);
values = magnitude(unique(cortex(:)));
assert(all(isfinite(values)), 'Cortical field contains undefined values.');
% Match ROAST's display convention; the upper tick explicitly marks saturation.
fieldMax = ceil(prctile(values, 95) * 100) / 100;
fprintf('Cortical |E| min/max: %.6g / %.6g V/m; color maximum %.2f V/m\n', min(values), max(values), fieldMax);

% Render at native 82 x 78 mm; TeX supplies all typography in the wrapper.
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [0 0 8.2 7.8]);
ax = axes(fig, 'Position', [0.015 0.25 0.97 0.68]);
patch(ax, 'Vertices', vertices, 'Faces', cortex, 'FaceVertexCData', magnitude, ...
    'FaceColor', 'interp', 'EdgeColor', 'none', 'FaceLighting', 'gouraud', ...
    'AmbientStrength', 0.65, 'DiffuseStrength', 0.35, 'SpecularStrength', 0);
hold(ax, 'on');
padColors = [0.75 0.22 0.14; 0.12 0.30 0.65];
for k = 1:2
    padFaces = mesh.face(mesh.face(:, 4) == 8 + k, 1:3);
    patch(ax, 'Vertices', vertices, 'Faces', padFaces, 'FaceColor', padColors(k, :), ...
        'EdgeColor', 'none', 'FaceLighting', 'gouraud');
end
axis(ax, 'equal', 'off');
view(ax, -15, 45);
camproj(ax, 'orthographic');
camzoom(ax, 1.35);
camlight(ax, 'headlight');
colormap(ax, parula(256));
caxis(ax, [0 fieldMax]);
bar = axes(fig, 'Position', [15/82 10/78 52/82 3/78]);
imagesc(bar, [0 fieldMax], [0 1], linspace(0, fieldMax, 256));
colormap(bar, parula(256));
set(bar, 'YDir', 'normal');
axis(bar, 'off');
set(fig, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 8.2 7.8], ...
    'PaperSize', [8.2 7.8], 'InvertHardcopy', 'off');
print(fig, fullfile(figureDir, 'stimulation_field_render.png'), '-dpng', '-r600');
close(fig);
fid = fopen(fullfile(figureDir, 'stimulation_field.tex'), 'w');
fprintf(fid, '%% Native size 82 x 78 mm; MNI152, pads 50 x 30 x 3 mm, padding20.\n');
fprintf(fid, '\\begin{tikzpicture}[thesis figure,x=1mm,y=1mm]\n');
fprintf(fid, '  \\node[anchor=south west,inner sep=0] at (0,0) {\\includegraphics{figures/02_problem_formulation/stimulation_field_render.png}};\n');
fprintf(fid, '  \\node at (22,75) {\\textcolor[rgb]{0.75,0.22,0.14}{TPP9: $+1$ mA}};\n');
fprintf(fid, '  \\node at (62,75) {\\textcolor[rgb]{0.12,0.30,0.65}{Ex8: $-1$ mA}};\n');
fprintf(fid, '  \\node at (41,17) {Electric field magnitude / $\\mathrm{V}\\,\\mathrm{m}^{-1}$};\n');
fprintf(fid, '  \\node[thesis secondary] at (15,6) {$0$};\n');
fprintf(fid, '  \\node[thesis secondary] at (41,6) {$%.3g$};\n', fieldMax / 2);
fprintf(fid, '  \\node[thesis secondary] at (67,6) {$\\geq %.2f$};\n', fieldMax);
fprintf(fid, '\\end{tikzpicture}\n');
fclose(fid);
close all;



% Build standalone PDF and a labeled PNG preview; keep build products in scratch.
wrapper = fullfile(scratch, 'figure_wrapper.tex');
writelines(["\documentclass[tikz,border=0pt]{standalone}", ...
    "\usepackage{XCharter}", "\usepackage{amsmath}", "\usepackage{graphicx}", ...
    "\tikzset{thesis figure/.style={font=\fontsize{9}{10.8}\selectfont},thesis secondary/.style={font=\fontsize{8}{9.6}\selectfont}}", ...
    "\begin{document}", "\input{figures/02_problem_formulation/stimulation_field}", ...
    "\end{document}"], wrapper);
cd(thesisDir);
[status, output] = system(sprintf('pdflatex -halt-on-error -interaction=nonstopmode -jobname=stimulation_field -output-directory="%s" "%s"', scratch, wrapper));
assert(status == 0, '%s', output);
copyfile(fullfile(scratch, 'stimulation_field.pdf'), fullfile(figureDir, 'stimulation_field.pdf'));
[status, output] = system(sprintf('pdftoppm -singlefile -png -r 300 "%s" "%s"', ...
    fullfile(figureDir, 'stimulation_field.pdf'), fullfile(figureDir, 'stimulation_field')));
assert(status == 0, '%s', output);
cd(previousDir);
clear restoreDir;
