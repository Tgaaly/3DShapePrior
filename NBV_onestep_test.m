function acc_eval = NBV_onestep_test(model, volume_size, pad_len, angle_inc, c)
% Next-Best-View one step test. Evaluate the recognition accuracies for
% different view seletion strategies: entropy-based, reconstruction-based,
% random selection, furthest away.

% volume_size: the size of volumetric representation(42).
% pad_len: padding size(3).
% angle_inc: rotation increment of each 3D mesh model(360).
% c: evaluate the class c(1 - 10).
rng('shuffle');
gpuDevice(1);

%[~, cmdout] = system('pbsnodes $HOST | egrep -o "[0-9]/${PBS_JOBID}"');
%gpuID = cmdout(1);
%gpuDevice(str2double(gpuID)+1)

if isempty(model)
    load ShapeNet.mat
end

classes = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};

data_path = 'NBV_data';

kernels;
index = 0;
test_label = zeros(1);

fprintf('testing on %s class\n', classes{c});
category_path = [data_path '/' classes{c}];
files = dir(category_path);

getCount = 0;
methods = zeros(10,4);
ent_eval = zeros(10,4);
acc_eval = zeros(10,4);
for i = 1 : length(files)
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
        continue;
    end
    if getCount == 10
        break;
    end
    filename = [category_path '/' files(i).name];
    index = index + 1;
    getCount = getCount + 1;
    test_label(index, 1) = c;
    tic;
    [this_prediction, prediction_next, NBV_v, RC_v, RAND_v, FUR_v] = NBV_onestep(model, filename, volume_size, pad_len, angle_inc);
    toc;
    
    num_v = size(prediction_next,1);
    
    entropy = zeros(num_v,1);
    for v = 1 : num_v
        temp = prediction_next(v,:);
        temp = temp(temp>0);
        entropy(v) = -sum(temp .* log(temp));
    end
   
    ent_eval(index,1) = entropy(NBV_v);
    ent_eval(index,2) = entropy(RC_v);
    ent_eval(index,3) = entropy(RAND_v);
    ent_eval(index,4) = entropy(FUR_v);
    
    [~ , predicted_label] = max(prediction_next, [], 2);
    acc_eval(index,1) = double(predicted_label(NBV_v) == c);
    acc_eval(index,2) = double(predicted_label(RC_v) == c);
    acc_eval(index,3) = double(predicted_label(RAND_v) == c);
    acc_eval(index,4) = double(predicted_label(FUR_v) == c);
    
    methods(index,1) = NBV_v;
    methods(index,2) = RC_v;
    methods(index,3) = RAND_v;
    methods(index,4) = FUR_v;
end

save(['res/acc_' classes{c}], ['acc_eval']);
save(['res/ent_' classes{c}], ['ent_eval']);
save(['res/med_' classes{c}], ['methods']);
