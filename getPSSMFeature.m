function  pssm_feature=getPSSMFeature(pssm_file)

%% 导入文本文件中的数据。
%% 初始化变量。
filename = pssm_file;
fileID = fopen(filename,'r');
aa=textscan(fileID,'%[^\n\r]');
startRow = 3;
endRow =size(aa{1,1},1)-5;
fclose(fileID);
formatSpec = '%*7s%6f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r');

%% 根据格式读取数据列。
% 该调用基于生成此代码所用的文件的结构。如果其他文件出现错误，请尝试通过导入工具重新生成代码。
textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% 关闭文本文件。
fclose(fileID);

%% 对无法导入的数据进行的后处理。
% 在导入过程中未应用无法导入的数据的规则，因此不包括后处理代码。要生成适用于无法导入的数据的代码，请在文件中选择无法导入的元胞，然后重新生成脚本。

%% 创建输出变量
tmpPSSM = [dataArray{1:end-1}];
%% 清除临时变量
clearvars filename startRow endRow formatSpec fileID dataArray ans;
pssm_feature=1./(ones(size(tmpPSSM))+exp(-tmpPSSM));
strpath=strcat(pssm_file,'_score.pssm');
dlmwrite(strpath, pssm_feature, 'delimiter', '\t');

end

