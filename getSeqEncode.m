function aa_data= getSeqEncode(seqfile)
[header,seq]=fastaread(seqfile);
len=size(seq,2);
if len>=700
    seqNEW=seq(1,1:700);
end
if len<700
    lenNEW=700-len;
    seqNEW=cat(2,seq,zeros(1,lenNEW));
end
aa_data=zeros(700,1);
for k=1:700
    c=seqNEW(1,k);
    if c == 'A'
        aa_data(k,1)=1;
    elseif c == 'R'
        aa_data(k,1)=2;
    elseif c == 'N'
        aa_data(k,1)=3;
    elseif c == 'D'
        aa_data(k,1)=4;
    elseif c == 'C'
        aa_data(k,1)=5;
    elseif c == 'Q'
        aa_data(k,1)=6;
    elseif c == 'E'
        aa_data(k,1)=7;
    elseif c == 'G'
        aa_data(k,1)=8;
    elseif c == 'H'
        aa_data(k,1)=9;
    elseif c == 'I'
        aa_data(k,1)=10;
    elseif c == 'L'
        aa_data(k,1)=11;
    elseif c == 'K'
        aa_data(k,1)=12;
    elseif c == 'N'
        aa_data(k,1)=13;
    elseif c == 'F'
        aa_data(k,1)=14;
    elseif c == 'P'
        aa_data(k,1)=15;
    elseif c == 'S'
        aa_data(k,1)=16;
    elseif c == 'T'
        aa_data(k,1)=17;
    elseif c == 'W'
        aa_data(k,1)=18;
    elseif c == 'Y'
        aa_data(k,1)=19;
    elseif c == 'V'
        aa_data(k,1)=20;
    end
end
outfilepath=strcat(seqfile,'_encode.txt');
dlmwrite(outfilepath,aa_data,'delimiter','\t')
end

