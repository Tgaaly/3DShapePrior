function remove_simplified_mesh(data_path)
% remove the over simplified graphics mesh model from the database

if ~exist('data_path','var')
    data_path = '/n/fs/modelnet/data';
end

classes = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'toilet', 'table'};

for c = 1 : length(classes)
    index = 0;
    fprintf('removing the %s class\n', classes{c});
    category_path = [data_path '/' classes{c} '/off'];
    bad_path = [data_path '/' classes{c} '/simplies'];
    if ~exist(bad_path, 'dir')
        mkdir(bad_path);
    end
    files = dir(category_path);
    
    num_vertices = [];
    num_polys = [];
    for i = 1 : length(files)
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [category_path '/' files(i).name];
        for viewpoint = 1 : 360/360
            if exist('axis', 'var')
                off_data = off_loader(filename, (viewpoint-1)*360, axis, stretch);
            else
                off_data = off_loader(filename, (viewpoint-1)*360);
            end
            index = index + 1;
            if size(off_data.vertices,1) < 100
                system(sprintf('mv %s %s',filename, bad_path));
            end
            num_vertices(index) = size(off_data.vertices , 1);
            num_polys(index) = size(off_data.faces, 1);
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
