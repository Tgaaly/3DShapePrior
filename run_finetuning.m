function run_finetuning()
% run generative finetuning

rng('shuffle');
gpuDevice(1);

%[~, cmdout] = system('pbsnodes $HOST | egrep -o "[0-9]/${PBS_JOBID}"');
%gpuID = cmdout(1);
%gpuDevice(str2double(gpuID)+1);

load pretrained_model.mat

num_layer = length(model.layers);
for l = 2 : num_layer
    model.layers{l}.grdw = zeros(size(model.layers{l}.w), 'single');
    model.layers{l}.grdb = zeros(size(model.layers{l}.b), 'single');
    model.layers{l}.grdc = zeros(size(model.layers{l}.c), 'single');
end

debug = true;
kernels;
volume_size = 42;
data_size = volume_size + 2 * 3;

base_path = './data';
data_list = read_data_list(base_path, data_size, 'train', debug);

% fine-tuning phase: wake-sleep algorithm
param = [];
param.epochs = 200;
param.lr = 0.00003;
param.momentum = [0.9, 0.9];
param.kCD = 3;
param.persistant = 1;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;

[model]= wake_sleep_CD(model, data_list, param);
    
save('finetuned_model','model');
