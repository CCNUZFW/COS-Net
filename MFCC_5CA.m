clc;clear;close all;clear all;
disp('Code start running...');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Step1:Extract MFCC. % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hi_train=zeros(300*60,256*44+1);
hi_test=zeros(216*60,256*44+1);

for n=1:60
    name={'Apple_air1','Apple_air2','Apple_air2(1)','honor_magic6pro','honor_V40','Honor9 STF-ALOO','huawei Novo2s',...
          'huawei_honor8(2)','huawei_honor10','huawei_M5','huawei_nova','huawei_nova3e','huawei_nova4e','huawei_p10',...
          'huawei_p20','huawei_TAG_AL00','huwei_honor7x','huwei_honor8','huwei_honor8(1)',...
          'huwei_honorV8','ipad7','iPad9','IPHONE_SE','IPHONE6','IPHONE6(1)',...
          'IPHONE6(2)','IPHONE6(3)','IPHONE6S','IPHONE6S(1)','IPHONE6S(2)',...
          'IPHONE7plus','iPhone12','iPhone13','iPhone13(1)','iPhone14','iPhone14(1)','IPHONEX','nubia_Z11','oppo_R9s','realme_GT2','realme_GT2(1)','Redmi 3S',...
          'Redmi Note4x','redmi_K40','redmi_note12pro_speed','SAMSUNG_S8','SPH-D710','vivo_X3F','vivo_x7',...
          'vivo_Y11t','xiaomi note3','xiaomi_13','xiaomi_mix2','xiaomi2s','xiaomi5',...
          'xiaomi8','xiaomi8se','xiaomi8se(1)','ZTE C880A','ZTE G719C'};
   disp(['Device: No.',num2str(n)]);
   for i=1:516   
     
       train_file=char(strcat('D:\ZYH\对比学习\CCNU_PLUS\',...
                 name(n),'\',name(n),'_',num2str(i),'.wav'));
       [s,fs]=audioread(train_file);
       
       [C, L] = wavedec(s, 5, 'haar'); %3层小波分解
        mfcc02=melcepst(s,fs,'EdD');
        % 提取每一层的逼近系数(低频)
        A1 = appcoef(C, L, 'haar', 1); %第一层
        A2 = appcoef(C, L, 'haar', 2); %第二层
        A3 = appcoef(C, L, 'haar', 3); %第三层
        A4 = appcoef(C, L, 'haar', 4); %第四层
        A5 = appcoef(C, L, 'haar', 5); %第五层
        
        % 降采样 cA 的时间分辨率
        mfcc_frames = size(mfcc02, 1); % MFCC 帧数
        A1_resampled = resample(A1, mfcc_frames, length(A1)); % 将 cA 调整到帧数一致
        A2_resampled = resample(A2, mfcc_frames, length(A2)); % 将 cA 调整到帧数一致
        A3_resampled = resample(A3, mfcc_frames, length(A3)); % 将 cA 调整到帧数一致
        A4_resampled = resample(A4, mfcc_frames, length(A4)); % 将 cA 调整到帧数一致
        A5_resampled = resample(A5, mfcc_frames, length(A5)); % 将 cA 调整到帧数一致
        
        %将mfcc与A1A2A3A4A5拼接
        combined_features = [mfcc02, A1_resampled,A2_resampled,A3_resampled,A4_resampled,A5_resampled]; % 拼接后维度为 1020×44
        num_rows = size(combined_features, 1);  % 1020
        num_selected = 256;  % 要选择的行数
        
        % 随机选择 256 行
        random_rows = randperm(num_rows, num_selected);
        
        % 使用随机选中的行生成新的矩阵
        final_features = combined_features(random_rows, :);
    
        hi_feature=reshape(final_features',1,256*44);
        
        if i<=300
           hi_train((n-1)*300+i,:)=[hi_feature,n-1];
        else
           hi_test((n-1)*216+i-300,:)=[hi_feature,n-1];
        end

   end
end

csvwrite('CCNUPLUS_train_300_256_MFCC+5cA_label.csv',hi_train);
csvwrite('CCNUPLUS_test_216_256_MFCC+5cA_label.csv',hi_test);
disp('MFCC extraction completed.');   