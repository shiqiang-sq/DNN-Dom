
#############################################################################
#Path setting
workpath=/home/shiqiang/DeepDomServer/temp/T00167156998
filepath="/home/shiqiang/DeepDomServer/temp/T00167156998/target.txt" # The target sequence
seqname=1nig_A

logpath=${filepath%.*}".log"
pssmpath=${filepath%.*}".pssm"
sspath=${filepath%.*}".ss"
accpath=${filepath%.*}".acc"
deepfeature=${filepath%.*}".deepfea"
rfsorepath=${filepath%.*}".rfscore"

#the scores that the residues belong to the domain boundary
probability_results=${filepath%.*}"_score.txt"

##the path of tools for extracting shallow features
#1) extracting ss(secondary structure) and acc (accessibility) by SCRATCH
SCRATCHpath="/home/shiqiang/SCRATCH-1D_1.0/bin/run_SCRATCH-1D_predictors.sh"
#2)extracting pssm by psiblast
psiblast_path="/home/shiqiang/bio/blast/bin/psiblast"
nr_path="/home/chujiner/bio/ncbi/db"
#3)matlab path
matlabpath="/home/shiqiang/MATLAB/R2017a/bin/matlab"

#the script for extracting deep feature, this is our deep model 
deepfeaturePath="/home/shiqiang/DeepDomServer/temp/getDeepFeatures3.py"

#the pBRF model for predicting domain boundaries 
pBRFpath="/home/shiqiang/DeepDomServer/temp/deepRF.py"

pythonpath="/opt/anaconda3/bin/python3"


cd  /home/shiqiang/DeepDomServer/temp
# the command lines for extracting pssm 
$psiblast_path -db nr -query ${filepath} -num_threads 50 -evalue 0.001 -save_each_pssm -save_pssm_after_last_round -outfmt 6 -max_target_seqs 9999999 -num_iterations 3 -out_ascii_pssm ${pssmpath} >>${logpath}
if [ -f ${pssmpath}".3" ];then
	pssmfile=${pssmpath}".3"
else
	pssmfile=${pssmpath}".2"

fi

# the command lines for extracting ss and acc
threads="50"
$SCRATCHpath ${filepath} ${filepath%.*} ${threads}


# the matlab scripts for processing pssm  ss  and acc
matlab -nodesktop -nosplash -r "getAllFeatures('${accpath}','${sspath}','${pssmfile}','${filepath}');quit;"
matlab -nodesktop -nosplash -r "getSeqEncode('${filepath}');quit;"

cd ${workpath}
## achieved the deep features by our deep model
${pythonpath} ${deepfeaturePath} ${filepath}".fea" ${filepath}"_encode.txt" ${deepfeature}

##achieved the scores of belonging to boundaries by p-BRF
${pythonpath} ${pBRFpath} ${deepfeature} ${sspath}"_score.ss3"  ${rfsorepath}

#achieved the predicted results by filtering the scores
cd  /home/shiqiang/DeepDomServer/temp
matlab -nodesktop -nosplash -r "getpredsfromscore('${rfsorepath}','${workpath}','${seqname}') ;quit;"

cd ${workpath}
probability_scores=`cat ${workpath}/targets.scores.txt`
domainboudanry=`cat ${workpath}/targets.result.txt`
