function data_list = read_off_list(base_path, train_or_test, debug)
% read raw off filenames

% base_path: root data folder
% train_or_test: select training data or testing data('train', 'test')
% debug: true for debug.

if debug
    maxNum = 20+2;
else
    maxNum = 200+2;
end

classes = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};
num_classes = length(classes);

data_list = cell(10,1);
for c = 1 : num_classes
    fprintf('reading the %s category\n', classes{c});
    category_path = [base_path '/' classes{c}  '/off/' train_or_test];
    files = dir(category_path);
    
    cat_ind = 0;
    data_list{c} = [];
    for i = 1 : min(length(files), maxNum)
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        filename = [category_path '/' files(i).name];
        
        cat_ind = cat_ind + 1;
        data_list{c}(cat_ind,1).filename = filename;
        data_list{c}(cat_ind,1).label = c;
    end
end

