function print_off(data_path, dest)
% visualize the off file in a volumetric representation and save the 2d
% image of the visualizetion.

addpath('../voxelization');

[ignore, category, ignore] = fileparts(data_path);

% for train
train_path = [data_path '/train'];
train_dest = [dest '/train'];
if ~exist(train_dest, 'dir')
    mkdir(train_dest);
end
files = dir(train_path);
fprintf('found %d instances of %s category 3D voxel data\n', length(files) - 3, category);

index = 0;
for i = 1 : length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || ~strcmp(files(i).name(end-2:end), 'off') ||files(i).isdir == 1
        continue;
    end
    index = index + 1;
    src_filename = [train_path '/' files(i).name];
    
    off_data = off_loader(src_filename, 0);
    instance = polygon2voxel(off_data, [30, 30, 30], 'auto');
    figure('visible','off') ;
    plot3D(instance); title([files(i).name]);
    view([-1,0.5,1])
    
    figure_name = [train_dest '/' files(i).name(1:end-3) 'png'];
    saveas(gcf, figure_name);
    close(gcf);
    
    if mod(index, 500) == 0
        fprintf('%d\n',index);
    elseif mod(index, 100) == 0
        fprintf('%d',index);
    elseif mod(index,10) == 0
        fprintf('.');
    end
end

% for test
test_path = [data_path '/test'];
test_dest = [dest '/test'];
if ~exist(test_dest, 'dir')
    mkdir(test_dest);
end
files = dir(test_path);
fprintf('found %d instances of %s category 3D voxel data\n', length(files) - 3, category);

index = 0;
for i = 1 : length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || ~strcmp(files(i).name(end-2:end), 'off') ||files(i).isdir == 1
        continue;
    end
    index = index + 1;
    src_filename = [test_path '/' files(i).name];
    
    off_data = off_loader(src_filename, 0);
    instance = polygon2voxel(off_data, [30, 30, 30], 'auto');
    figure('visible','off') ;
    plot3D(instance); title([files(i).name]);
    view([-1,0.5,1])
    
    figure_name = [test_dest '/' files(i).name(1:end-3) 'png'];
    saveas(gcf, figure_name);
    close(gcf);
    
    if mod(index, 500) == 0
        fprintf('%d\n',index);
    elseif mod(index, 100) == 0
        fprintf('%d',index);
    elseif mod(index,10) == 0
        fprintf('.');
    end
end


end

function offobj = off_loader(filename, theta)

offobj = struct();
fid = fopen(filename, 'rb');
OFF_sign = fscanf(fid, '%c', 3);
assert(strcmp(OFF_sign, 'OFF') == 1);

info = fscanf(fid, '%d', 3);
offobj.vertices = reshape(fscanf(fid, '%f', info(1)*3), 3, info(1))';
offobj.faces = reshape(fscanf(fid, '%d', info(2)*4), 4, info(2))';
if ~isempty(find(offobj.faces(:,1) == 4, 1))
    fprintf('nononononono\n');
end

% do some translation and rotation
center = (max(offobj.vertices) + min(offobj.vertices)) / 2;
offobj.vertices = bsxfun(@minus, offobj.vertices, center);
theta = theta * pi / 180;
R = [cos(theta), -sin(theta), 0;
     sin(theta), cos(theta) , 0;
        0      ,    0       , 1];

offobj.vertices = offobj.vertices * R;

% These vertices to define faces should be offset by one to follow the matlab convention.
offobj.faces = offobj.faces(:,2:end) + 1; 

fclose(fid);
end
