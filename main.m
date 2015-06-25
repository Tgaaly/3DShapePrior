clear, close all
format compact
dbstop if error

debug = 0;
kernels;
% setup_paths

run_pretrain(debug)

load('pretrained_model','model');

% test on training data
[predicted_label_tr] = rec_train(model,debug);

% run_finetuning(model,debug)

% load finetuned_model.mat
[predicted_label] = rec_test(model,debug);

 
% [samples] = sample_test_classification(model,2)
% show_sample(samples)