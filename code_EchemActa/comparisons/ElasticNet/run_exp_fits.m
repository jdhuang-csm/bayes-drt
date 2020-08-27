%% Set up files and tau bases
data_path = '../../data/experimental';
files = {'DRTtools_LIB_data.txt',...
    'DRTtools_LIB_data_qtr.csv',...
    'PDAC_COM3_02109_Contact10_2065C_500C.txt'};
taus = {(1./(2*pi*logspace(4,-5,200)))',...
    (1./(2*pi*logspace(4,-5,200)))',...
    (1./(2*pi*logspace(7,-3,200)))'
    };
suffixes = {'LIB_data','LIB_data_qtr','PDAC'};

%% Load and fit impedance files
for n = 1:length(files)
    file = files{n};
    disp(file)
    tau = taus{n};
    suffix = suffixes{n};
    filepath = strcat(data_path,'/',file);
    if strcmp(file,'DRTtools_LIB_data.txt')
        data = readtable(filepath,'Delimiter','\t');
        data.Properties.VariableNames{'Var1'} = 'Freq';
        data.Properties.VariableNames{'Var2'} = 'Zreal';
        data.Properties.VariableNames{'Var3'} = 'Zimag';
    elseif strcmp(file,'PDAC_COM3_02109_Contact10_2065C_500C.txt')
        data = readtable(filepath,'HeaderLines',22);
        data.Properties.VariableNames{'Var4'} = 'Freq';
        data.Properties.VariableNames{'Var5'} = 'Zreal';
        data.Properties.VariableNames{'Var6'} = 'Zimag';
    else
        data = readtable(filepath);
    end
    
    freq = data.Freq;
    A_file = strcat('A_',file(1:end-4),'.mat');
    [A_re,A_im] = get_A(A_file,freq,tau);
    
    % fit data
    [Z_res,g_res] = en_fit(data,tau,A_re,A_im,1,freq);
    % save results
    writetable(Z_res,strcat('results/Zout_',suffix,'.csv'))
    writetable(g_res,strcat('results/Gout_',suffix,'.csv'))
end

function [A_re,A_im] = get_A(A_file,f,tau)
    if isfile(A_file)
        A_load = load(A_file);
        A_re = A_load.A_re;
        A_im = A_load.A_im;
    else
        % if precomputed files not available, calculate matrices
        disp('Calculating A matrices...')
        [A_re,A_im] = cal_Basis(f,tau);
        save(A_file,'A_re','A_im')
        disp('Finished calculating matrices')
    end
end