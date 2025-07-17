clearvars;close all;clc;clear all;
%% System Model parameters
tic
kue = 4;
kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate
Pt=5; % in dBm
L =3; % number of channel paths (L)
% Note: The axes of the antennas match the axes of the ray-tracing scenario
My_ar=[8]; % number of LIS reflecting elements across the y axis
Mz_ar=[8]; % number of LIS reflecting elements across the z axis
M_bar=4; % number of active elements
K_DL=64; % number of subcarriers as input to the Deep Learning model
Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector
Training_Size = [2000];
% Preallocation of output variables
Rate_DLt=zeros(numel(My_ar),numel(Training_Size));
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));
params.scenario='O1_3p5B'; % DeepMIMO Dataset scenario: http://deepmimo.net/
params.active_BS=1; % active basestation(/s) in the chosen scenario
D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6/kue; % Bandwidth
Ut_row = 850; % user Ut row number
Ut_element = 1; % user Ut position from the row chosen above
Ur_rows = [600 900]; % user Ur rows
Validation_Size = 6200; % Validation dataset Size
K = 512; % number of subcarriers
miniBatchSize  = 500; % Size of the minibatch for the Deep Learning
% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of LIS reflecting elements across the x axis
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
% Generating the BF codebook
[BF_codebook]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
codebook_size=size(BF_codebook,2);
beta_min = 0.2;alpha = 1.5;phi = 0.43*180;
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

u_step=100;
Hdk = zeros(M,K_DL);
Hdxk = zeros(M,u_step);
for k = 1:1:kue
% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row;
params.active_user_last=Ut_row;
DeepMIMO_dataset=DeepMIMO_generator(params);
Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
Ht_abs = abs(Ht);
clear DeepMIMO_dataset
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
        Hdn = single(conj(DeepMIMO_dataset{1}.user{u}.channel));
    end
end
clear Delta_H
disp('=============================================================');
disp([' Calculating for M_bar = ' num2str(M_bar)]);
Rand_M_bar =unique(Rand_M_bar_all(1:M_bar)); % pick M_bar number of elements to be active
Ht_bar = reshape(Ht(Rand_M_bar,:),M_bar*K_DL,1);
DL_input = single(zeros(M_bar*K_DL*2,No_user_pairs));% 2 is real and imag parts, no_user_pairs is total sample points
DL_output = single(zeros(No_user_pairs,codebook_size));%predicted rate
DL_output_un=  single(zeros(numel(Validation_Ind),codebook_size));% Actual achievable rate
Delta_H_bar_max = single(0);
count=0;




for pp = 1:1:numel(Ur_rows_grid)-1
    %% Construct Deep Learning inputs
    Htx=repmat(Ht(:,1),1,u_step);
    Hdx=zeros(M,u_step);
    for u=1:u_step:params.num_user                        
        for uu=1:1:u_step
        Hd =single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel));  %Hd \in C^ N*M 
        Hdx(:,uu)=Hd(:,1);
        end
    end
end
Hdk(:,:,k)=1e-10.*Hd;
Hdxk(:,:,k)=1e-10.*Hdx;
end
toc