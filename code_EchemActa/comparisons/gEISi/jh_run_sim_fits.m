% fit all simulated files with single distribution

data_path = '../../../data/simulated/';

% DRT files
% drt_files = dir(strcat(data_path,'Z*.csv'));
% drt_files = drt_files(~contains({drt_files.name},'DDT'));
% fit_files(drt_files,@DRT,'DRT_modality.csv')

% TP-DDT files
% tp_files = dir(strcat(data_path,'Z*TP-DDT*.csv'));
% fit_files(tp_files,@transmissiveDDT,'TP-DDT_modality.csv')

% BP-DDT files
% bp_files = dir(strcat(data_path,'Z*BP-DDT*.csv'));
% fit_files(bp_files,@blockingDDT,'BP-DDT_modality.csv')

% DRT-TpDDT files
sp_files = dir(strcat(data_path,'Z_DRT-*-TpDDT*.csv'));
fit_files(sp_files,@jh_DRT_TpDDT,'DRT-TpDDT_modality.csv')

% DRT-TpDDT-BpDDT files (skip noiseless - too slow)
s2p_files = dir(strcat(data_path,'Z_DRT-TpDDT-BpDDT_uniform_0.25.csv'));
fit_files(s2p_files,@jh_DRT_TpDDT_BpDDT,'DRT-TpDDT-BpDDT_modality.csv')


function fit_files(files,fun,modality_file)
    functionHandle=functions(fun);
    if strcmp(functionHandle.function,'jh_DRT_TpDDT')
        modalities = zeros(length(files),2);
    elseif strcmp(functionHandle.function,'jh_DRT_TpDDT_BpDDT')
        modalities = zeros(length(files),3);
    else
        modalities = zeros(length(files),1);
    end
    for n = 1:length(files)
        file = files(n);
        disp(file.name)
        [modality,betak,Rml,muml,wml,tl,Fl,Z_res] = jh_fit_sim_file(file.name,fun,false);
        modalities(n,:) = modality;
        
        suffix = file.name(2:end);
        writetable(Z_res,strcat('results/Zout',suffix))
        
        % get predicted distribution(s)
        % Because tl may be of different lengths for different
        % distributions, must save each distribution in its own file
        FlTemp=Fl{1}; %DRT
        g_res = array2table([tl{1}' FlTemp(2,:)' FlTemp(1,:)' FlTemp(3,:)'],... 
            'VariableNames',{'tau' 'gamma' 'gamma_lo' 'gamma_hi'});
        writetable(g_res,strcat('results/Gout',suffix))
        
        if strcmp(functionHandle.function,'jh_DRT_TpDDT') || strcmp(functionHandle.function,'jh_DRT_TpDDT_BpDDT')
            Ftp=Fl{2}; %TP-DDT
            ftp_res = array2table([tl{2}' Ftp(2,:)' Ftp(1,:)' Ftp(3,:)'],... 
             'VariableNames',{'tau' 'ftp' 'ftp_lo' 'ftp_hi'});
            writetable(ftp_res,strcat('results/Ftp',suffix))
        end
        if strcmp(functionHandle.function,'jh_DRT_TpDDT_BpDDT')
            Fbp=Fl{3}; %BP-DDT
            fbp_res = array2table([tl{3}' Fbp(2,:)' Fbp(1,:)' Fbp(3,:)'],... 
             'VariableNames',{'tau' 'fbp' 'fbp_lo' 'fbp_hi'});
            writetable(fbp_res,strcat('results/Fbp',suffix))
        end
    end
    
    if strcmp(functionHandle.function,'jh_DRT_TpDDT')
        file_mod = table({files.name}', modalities(:,1), modalities(:,2),...
        'VariableNames',{'file' 'gamma_modality' 'ftp_modality'});
    elseif strcmp(functionHandle.function,'jh_DRT_TpDDT_BpDDT')
        file_mod = table({files.name}', modalities(:,1), modalities(:,2), modalities(:,3),...
        'VariableNames',{'file' 'gamma_modality' 'ftp_modality' 'fbp_modality'});
    else
        file_mod = table({files.name}', modalities,...
            'VariableNames',{'file' 'modality'});
    end
    writetable(file_mod,strcat('results/',modality_file));
end