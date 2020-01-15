function  pssm_feature=getPSSMFeature(pssm_file)

%% �����ı��ļ��е����ݡ�
%% ��ʼ��������
filename = pssm_file;
fileID = fopen(filename,'r');
aa=textscan(fileID,'%[^\n\r]');
startRow = 3;
endRow =size(aa{1,1},1)-5;
fclose(fileID);
formatSpec = '%*7s%6f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%[^\n\r]';

%% ���ı��ļ���
fileID = fopen(filename,'r');

%% ���ݸ�ʽ��ȡ�����С�
% �õ��û������ɴ˴������õ��ļ��Ľṹ����������ļ����ִ����볢��ͨ�����빤���������ɴ��롣
textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% �ر��ı��ļ���
fclose(fileID);

%% ���޷���������ݽ��еĺ���
% �ڵ��������δӦ���޷���������ݵĹ�����˲�����������롣Ҫ�����������޷���������ݵĴ��룬�����ļ���ѡ���޷������Ԫ����Ȼ���������ɽű���

%% �����������
tmpPSSM = [dataArray{1:end-1}];
%% �����ʱ����
clearvars filename startRow endRow formatSpec fileID dataArray ans;
pssm_feature=1./(ones(size(tmpPSSM))+exp(-tmpPSSM));
strpath=strcat(pssm_file,'_score.pssm');
dlmwrite(strpath, pssm_feature, 'delimiter', '\t');

end

