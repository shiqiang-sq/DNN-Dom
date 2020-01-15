function newfeature=getAllFeatures(accfile,ssfile,pssmfile,seqfile)
	acc_feature=getAccFeature(accfile);
	pssm_feature=getPSSMFeature(pssmfile);
	ss_feature=getSSFeature(ssfile);
	feature=cat(2,pssm_feature,ss_feature,acc_feature);
	len=size(feature,1);
	if len<700
		newfeature=zeros(700,25);
		newfeature(1:len,:)=feature;
	else
		newfeature=feature(1:700,:);		
	end
	
	outfilepath=strcat(seqfile,'.fea');
	dlmwrite(outfilepath,newfeature,'delimiter','\t')
end
