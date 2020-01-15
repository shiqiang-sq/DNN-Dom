function ss_feature=getSSFeature(ss_file)
     [HH,SS]=fastaread(ss_file);
     len=length(SS);
     ss_feature=[];
     for j=1:len
         c=SS(1,j);
         if c=='C'
             ss_feature=cat(1,ss_feature,[1 0 0]);
         end
         if c=='H'
             ss_feature=cat(1,ss_feature,[0 1 0]);
         end
         if c=='E'
             ss_feature=cat(1,ss_feature,[0 0 1]);
         end
     end
     strpath=strcat(ss_file,'_score.ss3');
     dlmwrite(strpath, ss_feature, 'delimiter', '\t');
end

