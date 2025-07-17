%clearvars;close all;clc;clear all;
tic
%% System Model parameters
kue = 4;
kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate
Pt=25; % in dBm
L =3; % number of channel paths (L)
% Note: The axes of the antennas match the axes of the ray-tracing scenario
My_ar=[4]; % number of RIS reflecting elements across the y axis
Mz_ar=[4]; % number of RIS reflecting elements across the z axis
K_DL=64; % number of subcarriers as input to the Deep Learning model
Training_Size = [50000];
% Preallocation of output variables
Rate_DLt=zeros(numel(My_ar),numel(Training_Size));
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));
params.scenario='O1_3p5B'; % DeepMIMO Dataset scenario: http://deepmimo.net/
params.active_BS=3; % active basestation(/s) in the chosen scenario
D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6/kue; % Bandwidth
Ut_row = 550; % user Ut row number
Ut_element = 1; % user Ut position from the row chosen above
Ur_rows = [600 900]; % user Ur rows
Validation_Size = 15000; % Validation dataset Size
K = 512; % number of subcarriers
miniBatchSize  = 500; % Size of the minibatch for the Deep Learning
% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of RIS reflecting elements across the x axis
My = My_ar(1);Mz = Mz_ar(1);
M = Mx.*My.*Mz; % Total number of LIS reflecting elements
% Preallocation of output variables
Rate_DL = zeros(1,length(Training_Size));
Rate_OPT = Rate_DL;
LastValidationRMSE = Rate_DL;
%--- Accounting SNR in ach rate calculations
%--- Definning Noisy channel measurements
Gt=3;             % dBi
Gr=3;             % dBi
NF=5;             % Noise figure at the User equipment
Process_Gain=10;  % Channel estimation processing gain
noise_power_dB=-204+10*log10(BW/K)+NF-Process_Gain; % Noise power in dB
SNR=10^(.1*(-noise_power_dB))*(10^(.1*(Gt+Gr+Pt)))^2; % Signal-to-noise ratio before considering path loss
SNR_dB = 10*log10(SNR);
% channel estimation noise
noise_power_bar=10^(.1*(noise_power_dB))/(10^(.1*(Gt+Gr+Pt)));
No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of (Ut,Ur) user pairs ??181????    all sample points      
RandP_all = randperm(No_user_pairs).'; % Random permutation of the available dataset
%% Starting the code
disp('======================================================================================================================');
disp([' Calculating for M = ' num2str(M)]);
Rand_M_bar_all = randperm(M); %random index number of RIS elements
   
%% Beamforming Codebook
% BF codebook parameters
over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
over_sampling_z=1;            % The beamsteering oversampling factor in the z direction
% Generating the active codebook
 Ntx = 4; Nty=4;
% [at_codebook]=sqrt(Ntx*Nty)*UPA_codebook_generator(1,Ntx,Nty,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
at_Nbeams = Ntx*Nty;
for i=1:1:at_Nbeams
    at_beams(i) = exp((-2*(at_Nbeams-1)*(i-1)*pi*1i)/at_Nbeams);
end
% Generating the passive codebook
[BF_codebook]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
codebook_size=size(BF_codebook,2);
beta_min = 0.8;alpha = 1.5;phi = 0.43*180;
BF_angles = asind(imag(BF_codebook)./1);
for i =1:Mx*My*Mz
    for j = 1:Mx*My*Mz
        if real(BF_codebook(i,j))<0
            BF_angles(i,j)=180-BF_angles(i,j);
        end
        if real(BF_codebook(i,j))>0 &&imag(BF_codebook(i,j))<0
            BF_angles(i,j)=BF_angles(i,j)+360;
        end
    end
end
amp =(1-beta_min)*(((sind(BF_angles-phi)+1)/2).^alpha)+beta_min;
BF_codebook = BF_codebook.*amp;
codebook_size = codebook_size*at_Nbeams;
%% DeepMIMO Dataset Generation
disp('-------------------------------------------------------------');
disp([' Calculating for K_DL = ' num2str(K_DL)]);          
% ------  Inputs to the DeepMIMO dataset generation code ------------ %
% Note: The axes of the antennas match the axes of the ray-tracing scenario
params.num_ant_x= Mx;             % Number of the UPA antenna array on the x-axis
params.num_ant_y= My;             % Number of the UPA antenna array on the y-axis
params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
params.ant_spacing=D_Lambda;          % ratio of the wavelnegth; for half wavelength enter .5        
params.bandwidth= BW*1e-9;            % The bandiwdth in GHz
params.num_OFDM= K;                   % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;        % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
params.saveDataset=0;
disp([' Calculating for L = ' num2str(params.num_paths)]);
% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row;
params.active_user_last=Ut_row;
DeepMIMO_dataset=DeepMIMO_generator(params);
Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
Ht_abs = abs(Ht);
clear DeepMIMO_dataset
sumrate = single(zeros(No_user_pairs,codebook_size));
input = zeros(16,64,2,54300);
input1 = zeros(16,64,1,54300);
input2 = zeros(16,64,1,54300);
for k = 1:1:kue
% ------------------ DeepMIMO "Ur" Dataset Generation -----------------%            
%Validation part for the actual achievable rate perf eval
Validation_Ind = RandP_all(end-Validation_Size+1:end);
[~,VI_sortind] = sort(Validation_Ind);
[~,VI_rev_sortind] = sort(VI_sortind);
%initialization
user_grids =[1.0 2751.0 181.0
2752.0 3852.0 181.0
3853.0 5203.0 361.0];
num_BS = 18;
Ur_rows_step = 100; % access the dataset 100 rows at a time
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2); % from 1000 to 1300 user grid (1000.1100,1200,1300)
Delta_H_max = single(0);
for pp = 1:1:numel(Ur_rows_grid)-1 % loop for Normalizing H (1,2,3)
    clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp); % Ur_rows_grid(1,2,3)= 1000,1100,1200
    params.active_user_last=Ur_rows_grid(pp+1)-1; % Ur_rows_grid(2,3,4)= 1099,1199,1299
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
    num_rows=max(min(user_grids(:,2),params.active_user_last)-max(user_grids(:,1),params.active_user_first)+1,0);
    params.num_user=sum(num_rows.*user_grids(:,3));                     % total number of users = 18100=181*100
    for u=1:params.num_user
        Hr = single(conj(DeepMIMO_dataset{1}.user{u}.channel));
        Delta_H = max(max(abs(Ht.*Hr+Hdk(:,:,k))));
        if Delta_H >= Delta_H_max
            Delta_H_max = single(Delta_H);
        end
    end
end
clear Delta_H
disp('=============================================================');
disp([' Calculating for M = ' num2str(M)]);
clear DL_input DL_output
DL_input = single(zeros(M*K_DL*2,No_user_pairs));% 2 is real and imag parts, no_user_pairs is total sample points
DL_output = single(zeros(No_user_pairs,codebook_size));%predicted rate
% DL_output_un=  single(zeros(numel(Validation_Ind),codebook_size));% Actual achievable rate
Delta_H_bar_max = single(0);
count=0;
for pp = 1:1:numel(Ur_rows_grid)-1
    %% Construct Deep Learning inputs
    u_step=100;
    Htx=repmat(Ht(:,1),1,u_step);
    Hrx=zeros(M,u_step);
%     jx = zeros(u_step,M,size(at_codebook,2));
    for u=1:u_step:params.num_user                        
        for uu=1:1:u_step
            Hr =single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel));  %Hr \in C^ N*M
            %--- Constructing the sampled channel
            n=sqrt(noise_power_bar/2)*(randn(M,K_DL)+1j*randn(M,K_DL));

            G = Hdk(:,:,k)+((Hr.*Ht).'*BF_codebook).'+n; 

            input1(:,:,:,u+uu-1+((pp-1)*params.num_user))= reshape(real(G) ,[16,64,1,1]); % first dimension: 2*256, second dimension : 18100
            input2(:,:,:,u+uu-1+((pp-1)*params.num_user))= reshape(imag(G) ,[16,64,1,1]); % first dimension: 2*256, second dimension : 18100

            
            Delta_H_bar = max(max(abs(G)));
            if Delta_H_bar >= Delta_H_bar_max
                Delta_H_bar_max = single(Delta_H_bar);
            end
            Hrx(:,uu)=Hr(:,1);
        end
        %--- Actual achievable rate for performance evaluation
        Hx = ((Htx.*Hrx).'*BF_codebook+Hdxk(:,:,k).');
        for i = 1:1:at_Nbeams
            jx(:,:,i) = Hx*at_beams(i);
        end
        Gx = reshape(jx,[u_step,M*at_Nbeams])+sqrt(noise_power_bar/2)*(randn(u_step,M*at_Nbeams)+1j*randn(u_step,M*at_Nbeams));
        SNR_sqrt_var = abs(Gx)./Delta_H_max;% power gain
        for uu=1:1:u_step
            if sum((Validation_Ind == u+uu-1+((pp-1)*params.num_user)))
                count=count+1;
                 % DL unnormalized output for validation
%                 DL_output_un(count,:) = single(sum(log2(1+(SNR*((SNR_sqrt_var(uu,:)).^2))),1));
            end
        end
        R = (log2(1+(SNR_sqrt_var).^2));
        % --- DL output normalization
        Delta_Out_max = max(R,[],2);
        if ~sum(Delta_Out_max == 0)
        Rn=diag(1./Delta_Out_max)*R;
        end
          DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = 1*Rn; %%%%% Normalized output, with each reflection%%%%%
    end
end
clear u Delta_H_bar Rn R jx Gx Hx SNR_sqrt_var
sumrate = sumrate+DL_output;
end
clear DL_output
%-- Sorting back the DL_output_un
% DL_output_un = DL_output_un(VI_rev_sortind,:);
%--- DL input normalization
input = cat(3,input1,input2);
input= 1*(input/Delta_H_bar_max); %%%%% Normalized from -1->1 %%%%%
%% DL Beamforming
%% ------------------ Beam selection -----------------%%
% DL_output_reshaped = reshape(DL_output.',1,1,size(DL_output,2),size(DL_output,1)); DLsize = size(DL_output_reshaped);
[maxrate,max_beam] = max(sumrate,[],2);
totalbeam = unique(max_beam);
averagerate = sum(maxrate)/size(sumrate,1);
% softdecision = zeros(size(sumrate,1),M*Ntx*Nty);
% sumsoftmax = sum(exp(sumrate),2);
% for i = 1:1:size(sumrate,2)
%     for j = 1:1:size(sumrate,1)
%     softdecision(j,i) = exp(sumrate(j,i))/sumsoftmax(i);
%     end
% end
% [~,max_decision] = max(softdecision,[],2);
% beam_unique = unique(max_decision);
A = zeros(M*Ntx*Nty,1);
B = zeros(M*Ntx*Nty,1);
%% ------------------ Training and Testing Datasets -----------------%%
% DL_output_reshaped_un = reshape(DL_output_un.',1,1,size(DL_output_un,2),size(DL_output_un,1));
% DL_output_un_szie = size(DL_output_reshaped_un);
for dd=1:1:numel(Training_Size)
    disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
    Training_Ind   = RandP_all(1:Training_Size(dd));
    %--------------------YTrain reshaping---------------------------%
    Y = (max_beam(Training_Ind,1));
%     beamusedy=unique(Y);
%     for j = 1:1:size(beamusedy)
%         B(beamusedy(j)) = beamusedy(j);
%     end
%     bny = find(~B);
%     YYYTrain = (cat(1,(Y),(bny)));
     beamYTrain = (tabulate(Y));
     YTrain = categorical(Y);  
    %-----------------------XTrain reshaping-----------------------%
    X = input(:,:,:,Training_Ind);
    %---------------------YValidation reshaping----------------------%
    YValidation = categorical(max_beam(Validation_Ind,1));
%     YVbeams = unique(YValidation);
%     for j = 1:1:size(YVbeams)
%         A(YVbeams(j)) = YVbeams(j);
%     end
%     bnyv = find(~A);
%     beamYVali = tabulate(cat(1,YValidation,bnyv));
%     YValidation = categorical(cat(1,YValidation,bnyv));
    %---------------------XValidation reshaping----------------------%
    XValidation = (input(:,:,:,Validation_Ind));
   %---------------------NN classess & weights----------------------%
    classes = categorical(beamYTrain(:,1));
     classweights = (beamYTrain(:,3));
   
%     Train = sortrows(cat(1,DL_input_reshaped,YYYTrain.').',K_DL*M*2+1);
%     YTrain = categorical(Train(:,K_DL*M*2+1));
%     XXTrain = Train(:,1:K_DL*M*2).';
%     XTrain= reshape(XXTrain,size(XXTrain,1),1,1,size(XXTrain,2));X_size = size(XTrain);
%     Validation = sortrows(cat(1,DL_val_reshaped,YValidation.').',K_DL*M*2+1);
%     YValidation = categorical(Validation(:,K_DL*M*2+1));
%     XXValidation = Validation(:,1:K_DL*M*2).';
%     XValidation= reshape(XXValidation,size(XXValidation,1),1,1,size(XXValidation,2));XVal_size = size(XValidation);
    clear YYTrain beamusedy bny YYYTrain max_decision XXValidation XXTrain Train Validation
    clear XVal maxrate max_beam DL_input_reshaped DL_val_reshaped DL_output_un
    clear VI_rev_sortind VI_sortind RandP_all
    layers = [
        imageInputLayer([16,64,2],'Name','input')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','relu1')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','relu2')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','relu3')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','relu4')
                
        
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','relu')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','rel')
        convolution2dLayer([3,3],2,'padding','same')
        reluLayer('Name','re')
                
        

        fullyConnectedLayer(M*at_Nbeams,'Name','Fully3')
        reluLayer('Name','relu5')
        dropoutLayer(0.5,'Name','dropout3')
        fullyConnectedLayer(M*at_Nbeams,'Name','Fully4')
        reluLayer('Name','relu6')
        dropoutLayer(0.5,'Name','dropout4')
        fullyConnectedLayer(M*at_Nbeams,'Name','Fully5')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','class1','Classes',classes,'ClassWeights',classweights)
        ];
    analyzeNetwork(layers)

    if Training_Size(dd) < miniBatchSize
        validationFrequency = Training_Size(dd);
    else
        validationFrequency = floor(Training_Size(dd)/miniBatchSize);
    end
    VerboseFrequency = validationFrequency;
    options = trainingOptions('sgdm', ... 
        'MiniBatchSize',200, ...
        'MaxEpochs',30, ...
        'InitialLearnRate',0.5, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',5, ...
        'L2Regularization',1e-4,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationFrequency',validationFrequency, ...
        'Plots','training-progress', ... % 'training-progress'
        'Verbose',0, ...    % 1  
        'ExecutionEnvironment', 'gpu', ...
        'VerboseFrequency',VerboseFrequency);
   
    % ------------- DL Model Training and Prediction -----------------%
% %     [~,Indmax_OPT]= max(YValidation,[],3);
% %     Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
% %     MaxR_OPT = single(zeros(numel(Indmax_OPT),1));                      
      [trainedNet,traininfo]  = trainNetwork(X,YTrain,layers,options);              
      YPredicted = predict(trainedNet,XValidation);  %predicted rate from trained netowrk

      
      %     % --------------------- Achievable Rate --------------------------%                    
%      [~,top10beams] = maxk(YPredicted,10,2);% predicted weight
%     validationrate = zeros(size(Validation_Ind,1),size(sumrate,2));
%     for i = 1:1:size(Validation_Ind)
%      validationrate(i,:) = sumrate(Validation_Ind(i),:);
%     end
%      [top10rate_bar,~] = maxk(validationrate,10,2); % wanted rate and beam
%     predictedrate = zeros(size(Validation_Ind,1),size(sumrate,2));
%      for i = 1:1:size(Validation_Ind)
%         for j = 1:1:10
%             predictedrate(i,j) = validationrate(i,top10beams(i,j));
%         end
%      end
%     overallloss =(predictedrate(:,1:10)-top10rate_bar(:,1:10)).^2;
%     rateloss1 =sum((overallloss(:,1))./(top10rate_bar(:,1)).^2)/size(Validation_Ind,1);
%     rateloss5 =sum(sum((overallloss(:,1:5))./(top10rate_bar(:,1:5)).^2))/size(Validation_Ind,1)*5;
%     rateloss10 =sum(sum((overallloss(:,1:10))./(top10rate_bar(:,1:10)).^2))/size(Validation_Ind,1)*10;
%    
%     clear predictedrate  top10beams YPredicted validationrate
%     beamcount = zeros(size(Validation_Ind,1),10);
%     for i = 1:1:size(Validation_Ind,1)
%         for j = 1:1:10
%             if overallloss(i,j)/(top10rate_bar(i,j).^2) < 0.01
%                 beamcount(i,j)=1;
%             end
%         end
%     end
%     top1beamacc = sum(beamcount(:,1))/size(Validation_Ind,1);
%     top5beam=zeros(size(Validation_Ind,1),1);
%     top10beam=zeros(size(Validation_Ind,1),1);
%     for i = 1:1:size(Validation_Ind,1)
%         if sum(beamcount(i,1:5),2)>0
%             top5beam(i)=1;
%         end
%         if sum(beamcount(i,1:10),2)>0
%             top10beam(i)=1;
%         end
%     end
%     top5beamacc = sum(top5beam,1)/size(Validation_Ind,1);
%     top10beamacc = sum(top10beam,1)/size(Validation_Ind,1);
%     clear beamcount top10beam top5beam beamcount overallloss top10rate_bar
% %     MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rates    
% %     for b=1:size(Indmax_DL,1)
% %         MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b))); %predicted value
% %         MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));  %Upper bound on achievable rates
% %     end
% %     Rate_OPT(dd) = mean(MaxR_OPT);          
% %     Rate_DL(dd) = mean(MaxR_DL);
% %     LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);                                          
% %     clear trainedNet traininfo YPredicted
% %     clear layers options Rate_DL_Temp MaxR_DL_Temp Highest_Rate
end           
toc