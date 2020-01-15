function acc_feature=getAccFeature(acc_file)
     [HH,SS]=fastaread(acc_file);
     len=length(SS);
     acc_feature=[];
     for j=1:len
         c=SS(1,j);
         if c=='e'
             acc_feature=cat(1,acc_feature,[1 0]);
         end
         if c=='-'
             acc_feature=cat(1,acc_feature,[0 1]);
         end
         if c=='b'
             acc_feature=cat(1,acc_feature,[0 1]);
         end
     end
     strpath=strcat(acc_file,'_score.acc');
     dlmwrite(strpath, acc_feature, 'delimiter', '\t');
end

