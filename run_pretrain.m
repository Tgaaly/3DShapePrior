function run_pretrain()
% Layer-wise pretraining.

rng('shuffle');
gpuDevice(1);

%[~, cmdout] = system('pbsnodes $HOST | egrep -o "[0-9]/${PBS_JOBID}"');
%gpuID = cmdout(1);
%gpuDevice(str2double(gpuID)+1)

debug = 0;

kernels;
volume_size = 42;
data_size = volume_size + 2 * 3;

% data path
base_path = './data';
data_list = read_data_list(base_path, data_size, 'train', debug);

param = [];
param.network = {
    struct('type', 'input');
    struct('type', 'convolution', 'outputMaps', 80, 'kernelSize', 6, 'actFun', 'sigmoid', 'stride', 3);
    struct('type', 'convolution', 'outputMaps', 320, 'kernelSize', 5, 'actFun', 'sigmoid', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 640, 'kernelSize', 4, 'actFun', 'sigmoid', 'stride', 1);
    struct('type', 'fullconnected', 'size', 3000, 'actFun', 'sigmoid');
    struct('type', 'fullconnected', 'size', 1200, 'actFun', 'sigmoid');
    struct('type', 'fullconnected', 'size', 800, 'actFun', 'sigmoid');
};

param.validation = 1;
param.classes = 10;
% This is to duplicate the labels for the final RBM in order to enforce the
% label training.
param.duplicate = 10;
param.data_size = [data_size, data_size, data_size, 1];

model = initialize_cdbn(param);

fprintf('\nmodel initialzation completed!\n\n');

%% train second layer (1st convolution) - see param.layer
tic;
param = [];
param.layer = 2;                            %layer idx
param.epochs = 150;                         %number of times going through training data
param.lr = 0.015;                           %learning rate
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;                             %CD-kPCD - size of markov chain in CD
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0.01;
param.sparse_cost = 0.03;
[model] = crbm2(model, data_list, param);
toc;

%% train 3rd and 4th convolutional layers
tic;
param = [];
param.layer = 3;
param.epochs = 400;
param.lr = 0.003;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0.05;
param.sparse_cost = 0.1;
[model] = crbm(model, data_list, param);
toc;

%% train 4th convolutional layer
tic;
param = [];
param.layer = 4;
param.epochs = 600;
param.lr = 0.002;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
[model] = crbm(model, data_list, param);
toc;

%% feed-forard
[hidden_prob_h4, train_label] = propagate_data(model, data_list, 5);

%% first fully connected layer (5th layer)
tic;
param = [];
param.layer = 5;
param.epochs = 600;
param.lr = 0.002;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
hidden_prob_h4 = reshape(hidden_prob_h4, size(hidden_prob_h4,1),[]);
[model, hidden_prob_h5] = rbm(model, hidden_prob_h4, param);
toc;

%% train fully connected rbm just before last layer rbm (6th layer)
tic;
param = [];
param.layer = 6;
param.epochs = 1000;
param.lr = 0.005;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
hidden_prob_h5 = reshape(hidden_prob_h5, size(hidden_prob_h5,1),[]);
[model, hidden_prob_h6] = rbm(model, hidden_prob_h5, param);
toc;

%% train last layer (output layer)
tic;
param = [];
param.layer = 7;
param.epochs = 2000;
param.lr = 0.0003;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 1;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
[model] = rbm_last(model, [train_label, hidden_prob_h6], param);
toc;

%% removing computed gradients from the layers
for l = 2 : length(model.layers)
    model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdb');
    model.layers{l} = rmfield(model.layers{l},'grdc');
end

%% saving pretrained model
save('pretrained_model','model');
