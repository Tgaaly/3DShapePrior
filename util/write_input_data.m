function write_input_data(data_path, volume_size, pad_len, angle_inc)
% Put the mesh object in a volume grid and save the volumetric
% represenation file.

addpath ../voxelization;

classes = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};
data_size = pad_len * 2 + volume_size;
for c = 1 : length(classes)
    fprintf('writing the %s category\n', classes{c});
    category_path = [data_path '/' classes{c} '/off'];
    dest_path = [data_path '/' classes{c} '/' num2str(data_size)];
    if ~exist(dest_path, 'dir')
        mkdir(dest_path);
    end
    % for train
    train_path = [category_path '/train'];
    dest_final_path = [dest_path '/train'];
    if ~exist(dest_final_path, 'dir')
        mkdir(dest_final_path);
    end
    files = dir(train_path);
    for i = 1 : length(files)     
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [train_path '/' files(i).name];
        for viewpoint = 1 : 360/angle_inc
            destname = [dest_final_path '/' files(i).name(1:end-4) '_' num2str(viewpoint) '.mat'];
            if exist('axis', 'var')
                off_data = off_loader(filename, (viewpoint-1)*angle_inc, axis, stretch);
            else
                off_data = off_loader(filename, (viewpoint-1)*angle_inc);
            end
            instance = polygon2voxel(off_data, [volume_size, volume_size, volume_size], 'auto');
            instance = padarray(instance, [pad_len, pad_len, pad_len]);
            instance = int8(instance);
            save(destname, 'instance');
        end
    end
    
    % for test
    test_path = [category_path '/test'];
    dest_final_path = [dest_path '/test'];
    if ~exist(dest_final_path, 'dir')
        mkdir(dest_final_path);
    end
    files = dir(test_path);
    for i = 1 : length(files)     
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [test_path '/' files(i).name];
        for viewpoint = 1 : 360/angle_inc
            destname = [dest_final_path '/' files(i).name(1:end-4) '_' num2str(viewpoint) '.mat'];
            if exist('axis', 'var')
                off_data = off_loader(filename, (viewpoint-1)*angle_inc, axis, stretch);
            else
                off_data = off_loader(filename, (viewpoint-1)*angle_inc);
            end
            instance = polygon2voxel(off_data, [volume_size, volume_size, volume_size], 'auto');
            instance = padarray(instance, [pad_len, pad_len, pad_len]);
            instance = int8(instance);
            save(destname, 'instance');
        end
    end
end

function offobj = off_loader(filename, theta, axis, stretch)

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
if exist('axis', 'var')
    switch axis
        case 'x',
            offobj.vertices(:,1) = offobj.vertices(:,1) * stretch;
        case 'y',
            offobj.vertices(:,2) = offobj.vertices(:,2) * stretch;
        case 'z',
            offobj.vertices(:,3) = offobj.vertices(:,3) * stretch;
        otherwise,
            error('off_loader axis set wrong');
    end
end
theta = theta * pi / 180;
R = [cos(theta), -sin(theta), 0;
     sin(theta), cos(theta) , 0;
        0      ,    0       , 1];

offobj.vertices = offobj.vertices * R;

% These vertices to define faces should be offset by one to follow the matlab convention.
offobj.faces = offobj.faces(:,2:end) + 1; 

fclose(fid);