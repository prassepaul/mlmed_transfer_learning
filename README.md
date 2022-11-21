# Code for "Pre-training on In-Vitro and Fine-Tuning on Patient-Derived Data Improves Neural Drug Sensitivity Models"
This repository contains the code for the paper "Pretraining on In-Vitro and Fine-Tuning on Patient-Derived Data Improves Neural Drug Sensitivity Models for Precision Oncology". The paper can be found here: https://www.mdpi.com/2072-6694/14/16/3950


## Data Creation
To run all the experiments, you need to download the data for the datasets.

### Beat-AML
The data used for the experiments on the Beat-AML dataset can be found here: https://www.synapse.org/#!Synapse:syn20940518/wiki/600156.
The data should be stored in the folder data/beat_aml:
    * beat_aml_rnaseq.csv (rnaseq.csv)
    * beat_aml_aucs.csv   (aucs.csv)
    
Create the data with the notebook: data_creation_data_visualization.ipynb

### CCLE
The data used for the experiments on the CCLE dataset can be found here:
Download the file https://data.broadinstitute.org/ccle_legacy_data/cell_line_annotations/CCLE_sample_info_file_2012-10-18.txt into data/ccle.
Download the file https://data.broadinstitute.org/ccle_legacy_data/pharmacological_profiling/CCLE_NP24.2009_Drug_data_2015.02.24.csv into data/ccle.
Download the file https://data.broadinstitute.org/ccle_legacy_data/pharmacological_profiling/CCLE_NP24.2009_profiling_2012.02.20.csv into data/ccle.
Download the file https://data.broadinstitute.org/ccle/CCLE_DepMap_18q3_RNAseq_RPKM_20180718.gct into data/ccle.

Create the data with the notebook: data_creation_data_visualization.ipynb

### PDO
The data used for the experiments on the PDO dataset can be found here: https://aacrjournals.org/cancerdiscovery/article/8/9/1112/10165/Organoid-Profiling-Identifies-Common-Responders-to.
Download the file https://aacr.silverchair-cdn.com/aacr/content_public/journal/cancerdiscovery/8/9/10.1158_2159-8290.cd-18-0349/5/21598290cd180349-sup-199398_2_supp_4775187_p95dln.xlsx into data/organoid_pancreas.
Download the file https://aacr.silverchair-cdn.com/aacr/content_public/journal/cancerdiscovery/8/9/10.1158_2159-8290.cd-18-0349/5/21598290cd180349-sup-199398_2_supp_4775186_p95dln.xlsx into data/organoid_pancreas.

Create the data with the notebook: data_creation_data_visualization.ipynb

### Xenografts
The data can be found in data/lung_cancer_xenografts

### GDSC Data
The data used in the experiments on the GDSC dataset can be found here:ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/.
Download the file ftp://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/GDSC1_fitted_dose_response_25Feb20.xlsx to data/.
Doanload and extract the file https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip to data/.

### Create drug-descriptions
Run create_descriptors_for_smiles.ipynb to create the descriptions for the drugs.


### Run all the experiments
Run *.sh scripts

### Plot results
Run plot_print_results_precision_oncology_drug_development_only_pretrain to plot/reproduce the results presented in the paper.
