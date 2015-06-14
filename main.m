clear, close all
format compact
dbstop if error

debug = 0;

%% visualize test
% param = [];
% param.debug = debug;
% param.volume_size = 21;%24;
% param.pad_size = 0;%3;
% data_size = param.volume_size + 2 * param.pad_size;
% param.data_path = '../Data/data_CDBM_cl2';
% param.classnames = {'good','bad'};
% param.classes = length(param.classnames);
% data_list = read_data_list(param.data_path, param.classnames, data_size, 'train', param.debug);
% load([data_list{1}(200).filename])%../Data/data_CDBM_cl2/good/21/train/good_000001_1.mat
% Volume=logical(instance);
% Volume2 = imfill(Volume,'holes');
% figure
% [X,Y,Z]=ind2sub(size(Volume),find(Volume(:)));
% plot3(X,Y,Z,'b.'), hold on
% [X,Y,Z]=ind2sub(size(Volume2),find(Volume2(:)));
% plot3(X,Y,Z,'r.');
% axis equal;
% xlabel('x');
% ylabel('y');
% zlabel('z');

%% old code
run_pretrain_old()

load('pretrained_model','model');
run_finetuning_old()

load finetuned_model.mat
[predicted_label] = rec_test_old(model);

%%
run_pretrain(debug)

load('pretrained_model','model');
run_finetuning(model,debug)

load finetuned_model.mat
[predicted_label] = rec_test(model,debug);

 
[samples] = sample_test_classification(model,1)
show_sample(samples)
