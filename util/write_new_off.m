function [data] = write_new_off(data_path, dest)
% rotate the mesh object to the correct pose. The rotation angle should be
% manually specified. This function is just used when I manually align all
% the data.

addpath('../voxelization');

[ignore, category, ignore] = fileparts(data_path);
files = dir(data_path);
fprintf('found %d instances of %s category 3D voxel data\n', length(files) - 3, category);

index = 0;
for i = 1 : length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end),'off')
        continue;
    end
    index = index + 1;
    src_filename = [data_path '/' files(i).name];
    dest_filename = [dest '/' files(i).name];
    off_rotate(src_filename, dest_filename, 90);
    
    %off_data = off_loader(dest_filename, 0);
    %instance = polygon2voxel(off_data, [30, 30, 30], 'auto');
    %plot3D(instance); title([files(i).name]);
    if mod(index, 500) == 0
        fprintf('%d\n',index);
    elseif mod(index, 100) == 0
        fprintf('%d',index);
    elseif mod(index,10) == 0
        fprintf('.');
    end
end


end
function off_rotate(src_filename, dest_filename, theta)
    fid = fopen(src_filename, 'rb');
    OFF_sign = fscanf(fid, '%c', 3);
    assert(strcmp(OFF_sign, 'OFF') == 1);

    wid = fopen(dest_filename, 'wb');
    count = fprintf(wid, '%c', OFF_sign);
    count = fprintf(wid, '\n');
    
    info = fscanf(fid, '%d', 3);
    count = fprintf(wid, '%d %d %d\n', info);
    
    vertices = reshape(fscanf(fid, '%f', info(1)*3), 3, info(1))';
    faces = reshape(fscanf(fid, '%d', info(2)*4), 4, info(2))';
    if ~isempty(find(faces(:,1) == 4, 1))
        fprintf('nononononono\n');
    end

    % do some translation and rotation
    center = (max(vertices) + min(vertices)) / 2;
    vertices = bsxfun(@minus, vertices, center);
    theta = theta * pi / 180;
    R = [cos(theta), -sin(theta), 0;
         sin(theta), cos(theta) , 0;
            0      ,    0       , 1];

    vertices = vertices * R;
    for i = 1 : info(1)
        count = fprintf(wid, '%f %f %f\n', vertices(i,:));
    end
    for i = 1 : info(2)
        count = fprintf(wid, '%d %d %d %d\n', faces(i,:));
    end
     
    fclose(wid);
    fclose(fid);
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
