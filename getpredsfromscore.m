function getpredsfromscore(filename,rootpath,seqname)
    th=0.61
    for JJ=1:1
        %filename=strcat('/home/shiqiang/DeepDomServer/temp/','rfscore0.txt');
        [score] =getScores( filename );
        ss=score(:,2);
        ss=smooth(ss,30);
        filterdscorePath=strcat(rootpath,'/targets.scores.txt');
        dlmwrite(filterdscorePath,ss','delimiter','|');
        len=size(ss,1);
        [Hx,Wx]=find(ss>th);
        label=2*ones(len,1);
        label(Hx,1)=1;
        newFilePath=strcat(rootpath,'targets.lable');
        dlmwrite(newFilePath,label,'delimiter','\t');
        predDataPath=strcat(rootpath,'/targets.result.txt');
        [domainLinkers_post] = getDomainLinkerNEW2(label,seqname,predDataPath,ss);
    end
end