%%%
% 
%  Speaker Recognition based on Gaussian Mixture Model  
%
%  for Classification and Recognition 2012/2013
%  by Jakub Vojvoda, vojvoda@swdeveloper.sk
%  
%%%

addpath('lib');

% training data
d1 = 'train/';
d2 = 'dev/';

% reading the training data
for i= 1 : 31
  dir_name1 = strcat(d1, num2str(i), '/*.wav');
  dir_name2 = strcat(d2, num2str(i), '/*.wav');
  files = [dir([dir_name1]); dir([dir_name2])];
  
  for j= 1 : length(files)
    try
      f = strcat(d1, num2str(i), '/', files(j).name);
      wav_data{i}{j} = wavread([f]);
      disp(f);
    catch err
      f = strcat(d2, num2str(i), '/', files(j).name);
      wav_data{i}{j} = wavread([f]);
      disp(f);
    end;   
  end;
end;

% data pre-processing 
crop = 8000;
frame_length = 2400;
threshold = 0.01;
x = 1;

for i= 1 : length(wav_data)
  for j= 1 : length(wav_data{i})
    
    % acustic artefact removing
    wav_data{i}{j} = wav_data{i}{j}(crop : (length(wav_data{i}{j}) - crop + frame_length));
    
    % silence removing
    N = length(wav_data{i}{j});
    frame_count = floor(N / frame_length);
    
    for k= 1 : frame_count
      frame = wav_data{i}{j}((k-1)*frame_length + 1 : frame_length * k);
      max_range = max(frame);
      
      % thresholding
      if (max_range > threshold)
        signals{i}{j}((x-1)*frame_length + 1 : frame_length * x) = frame;
        x = x + 1;
        clear frame;
      end;
    end;
    %plot(signals{i}{j});
    %pause;      
    x = 1;
  end;
end;

% MFCC features extraction
for i= 1 : 31
  for j= 1 : length(signals{i})
    features{j} = mfcc(signals{i}{j}, 400, 240, 512, 16000, 23, 13);
  end;
  train_data{i} = cell2mat(features);
end;

% number of training iterations
train_count = 8;

for i= 1 : 31
  % number of Gaussian Mixtures
  gauss_count{i} = 5;
  
  % expected value calculation 
  mtd = train_data{i};
  x = floor(length(mtd)/4); 
  y = floor(length(mtd)/2);
  z = floor((3 * length(mtd))/4);
  
  for n= 1 : 13
    mean_vector(n,1) = mean(mean(mtd(n,:)));
    mean_vector(n,2) = mean(mean(mtd(n,(1:x))));
    mean_vector(n,3) = mean(mean(mtd(n,(z:end))));
    mean_vector(n,4) = mean(mean(mtd(n,(x:y))));
    mean_vector(n,5) = mean(mean(mtd(n,(y:z))));
  end;
  
  gauss_meanv{i} = mean_vector;  

  % covariance matrix
  gauss_covm{i} = repmat(var(train_data{i}', 1)', 1, gauss_count{i});

  % weighting vector
  gauss_weight{i} = ones(1, gauss_count{i}) / gauss_count{i};      
  
  % training process
  for j=1 : train_count
    [gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}, ttl{i}] = train_gmm(train_data{i}, gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}); 
  end;
  
  disp(['Train data: ' num2str(i)])
  disp(['  Total log-likelihood: ' num2str(ttl{i})])
end;

% data writing
d = 'eval';
clear f;

% directory reading
test_dir_name = strcat(d, '/*.wav');
test_files = dir([test_dir_name]);

for i= 1 : length(test_files)
  f = strcat(d, '/', test_files(i).name);
  disp(f);
  test_wav_data{i} = wavread([f]);
  [file_name{i} fff]= regexp(test_files(i).name, '\.', 'split');   
end;    

crop = 8000;
frame_length = 2400;
threshold = 0.01;
x = 1;

% silent frames removing
for i= 1 : length(test_wav_data)
  test_wav_data{i} = test_wav_data{i}(crop : (length(test_wav_data{i}) - crop + frame_length));
    
  N = length(test_wav_data{i});
  frame_count = floor(N / frame_length);
    
  for k= 1 : frame_count
    frame = test_wav_data{i}((k-1)*frame_length + 1 : frame_length * k);
    max_range = max(frame);
      
    if (max_range > threshold)
      test_signals{i}((x-1)*frame_length + 1 : frame_length * x) = frame;
      x = x + 1;
      clear frame;
    end;
  end;
  x = 1;
end;

% MFCC feature extraction
for i= 1 : length(test_signals)
  test_data{i} = mfcc(test_signals{i}, 400, 240, 512, 16000, 23, 13);
end;

% creating a file
fileID = fopen('audio_GMM.txt', 'at');
formatSpec = '%s %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
  
for i= 1 : length(test_data)
  for k= 1 : 31
    % class evaluation
    ll(k) = sum(logpdf_gmm(test_data{i}, gauss_weight{k}, gauss_meanv{k}, gauss_covm{k}));
  end;
  % final class decision  
  [maximum_ll, index] = max(ll);
  
  fprintf(fileID,formatSpec, file_name{i}{1},index,ll(1),ll(2),ll(3),ll(4),ll(5),ll(6),ll(7),ll(8),ll(9),ll(10),ll(11),ll(12),ll(13),ll(14),ll(15),ll(16),ll(17),ll(18),ll(19),ll(20),ll(21),ll(22),ll(23),ll(24),ll(25),ll(26),ll(27),ll(28),ll(29),ll(30),ll(31));                
end;                                  

fclose(fileID);
