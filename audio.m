clc;
close all;

addpath('/home/asthemus/speaker-recognition/lib');
global_path = '/home/asthemus/speaker-recognition/lib/';
%x1 = audioread('/home/asthemus/speaker-recognition/lib/train/id10001/0001.wav');
%x1 = transpose(x1);
%global_data{1}{1}  = x1;
%x2 = audioread('/train/id10001/00002.wav');
%x2 = transpose(x2);
%global_data{1}{2} = x2;
%x3 = audioread('/train/id10001/00003.wav');
%x3 = transpose(x3);
%global_data{1}{3} = x3;
%f=[];
%f = [f, x1];
%f = [f, x2];
%f = [f, x3];
%global_f{1} = f;


%x4 = audioread('/train/id10001/0004.wav');
%x4 = transpose(x4);
%x5 = audioread('/train/id10001/00004.wav');
%x5 = transpose(x5);
%global_data{2}{1} = x4;
%global_data{2}{2} = x5;
%g=[];
%g = [g, x4];
%g = [g, x5];
%global_f{2} = g;

%% training data load
dir_start  = 'id1';

for i=1:10   
     id_padded = sprintf('%04d',i);
     dir_global = strcat(global_path,'train/');
     dir_name = strcat(dir_start, id_padded);
     disp(strcat('checking directory ', dir_name));
     D = strcat(dir_global,dir_name);
     sub_files = dir(D);
     
     for k=1:length(sub_files)
        file_names=setdiff({sub_files(k).name},{'.','..'});
        file_loc = strcat(D,'/',file_names);
        disp(file_loc);
        user_audio_file=[];
        if(~isempty(file_loc))
            user_audio_file = audioread(file_loc{1});
            global_data{i}{k} = user_audio_file;
        end    
     end
     
     global_data{i} = global_data{i}(3:end);
end

%% training data processing

    
% data pre-processing 
crop = 8000;
frame_length = 2400;
threshold = 0.01;
l=22;
Fs = 44100/4;
R = [300 3700];
alpha=0.97;
nceps =13;
x = 1;

N_GMM = 16;     % num of GMM classes
        

for i= 1 : length(global_data)
    disp(strcat('pre-processing for id ',int2str(i)));
  for j= 1 : length(global_data{i})
    
    % acustic artefact removing
    global_data{i}{j} = global_data{i}{j}(crop : (length(global_data{i}{j}) - crop + frame_length));
    
    % silence removing
    N = length(global_data{i}{j});
    frame_count = floor(N / frame_length);
    
    for k= 1 : frame_count
      frame = global_data{i}{j}((k-1)*frame_length + 1 : frame_length * k);
      max_range = max(frame);
      
      % thresholding
      if (max_range > threshold)
        signals{i}{j}((x-1)*frame_length + 1 : frame_length * x) = frame;
        x = x + 1;
        clear frame;
      end
    end
    %plot(signals{i}{j});
    %pause;      
    x = 1;
  end
end
    
%MFCC features extraction
for i= 1 : 10
    disp(strcat('feature extranction for id', int2str(i)));
  for j= 1 : length(signals{i})
    [features{j},~,~] = mfcc(signals{i}{j}, 400, 240, Fs, 23, l, nceps, hamming(64), R, alpha);
  end
  train_data{i} = cell2mat(features);
end

% After getting the feature, we train a GMM model, N_GMM

%GMM Training
tic
options = statset('MaxIter', 500);
for i=1:10
    disp(strcat('training GMM model for id ',int2str(i)));
    GMModel{i} = fitgmdist(transpose(train_data{1}), N_GMM, 'Options', options);
end
toc


%% number of training iterations
train_count = 8;

%for i= 1 : 10
%number of Gaussian Mixtures
  %gauss_count{i} = 5;
  
  % expected value calculation 
  %mtd = train_data{i};
  %x = floor(length(mtd)/4); 
  %y = floor(length(mtd)/2);
  %z = floor((3 * length(mtd))/4);
  
  %for n= 1 : nceps
  %  mean_vector(n,1) = mean(mean(mtd(n,:)));
  %  mean_vector(n,2) = mean(mean(mtd(n,(1:x))));
  %  mean_vector(n,3) = mean(mean(mtd(n,(z:end))));
  %  mean_vector(n,4) = mean(mean(mtd(n,(x:y))));
  %  mean_vector(n,5) = mean(mean(mtd(n,(y:z))));
  %end
  
  %n_dim = size(data, 2);
  
  %gauss_meanv{i} = mean_vector;  

  % covariance matrix
  %gauss_covm{i} = zeros(n_dim,n_dim,gauss_count{i});

  % weighting vector
  %gauss_weight{i} = ones(1, gauss_count{i}) / gauss_count{i};      
  
  % training process
%   for j=1 : train_count
%     [gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}, ttl{i}] = train_gmm(train_data{i}, gauss_weight{i}, gauss_meanv{i}, gauss_covm{i}); 
%   end
%   
%   disp(['Train data: ' num2str(i)])
%   disp(['  Total log-likelihood: ' num2str(ttl{i})])
%end

