function data = load_data(data_dir, data_type)
    if ~(strcmp(data_type, 'video') || strcmp(data_type, 'audio'))
        error('Data type should be either audio or video');
    end

    train_data = cell(850, 1);
    train_label = zeros(850, 10);
    valid_data = cell(50, 1);
    valid_label = zeros(50, 10);
    test_data = cell(150, 1);
    test_label = zeros(150, 10);
    
    train_ptr = 1;
    valid_ptr = 1;
    cur_dir = fullfile(data_dir, 'train', data_type);
    data_files = dir(cur_dir);
    for idx = 1:size(data_files, 1)
        if data_files(idx, 1).isdir == 0
            file_name = data_files(idx, 1).name;
            spk_id = str2double(file_name(2:3));
            cur_digit = str2double(file_name(6));
            if spk_id < 21
                temp= load(fullfile(cur_dir, file_name));
                train_data{train_ptr, 1} = temp.data;
                train_label(train_ptr, cur_digit+1) = 1;
                train_ptr = train_ptr + 1;
            else
                temp= load(fullfile(cur_dir, file_name));
                valid_data{valid_ptr, 1} = temp.data;
                valid_label(valid_ptr, cur_digit+1) = 1;
                valid_ptr = valid_ptr + 1;
            end
        end
    end
    
    test_ptr = 1;
    cur_dir = fullfile(data_dir, 'test', data_type);
    data_files = dir(cur_dir);
    for idx = 1:size(data_files, 1)
        if data_files(idx, 1).isdir == 0
            file_name = data_files(idx, 1).name;
            cur_digit = str2double(file_name(6));
            
            temp= load(fullfile(cur_dir, file_name));
            test_data{test_ptr, 1} = temp.data;
            test_label(test_ptr, cur_digit+1) = 1;
            test_ptr = test_ptr + 1;
        end
    end
    
    data.train_data = train_data;
    data.train_label = train_label;
    data.valid_data = valid_data;
    data.valid_label = valid_label;
    data.test_data = test_data;
    data.test_label = test_label;
end