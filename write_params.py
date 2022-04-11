import json

param_dict = {   'gdsc_rma_data_path':'data/Cell_line_RMA_proc_basalExp.txt',
                 'beat_rna_data_path':'data/beat_aml/beat_aml_rnaseq.csv',
                 'xenografts_rna_data_path':'data/lung_cancer_xenografts/normalized_expression.remoatFiltered.geneLevel.txt',
                 'ccle_rna_data_path':'data/ccle/rpkm_gene_data.csv',
                 'pancreas_rna_data_path':'data/organoid_pancreas/organoid_pancreas_fpkm.txt',
                 'gdsc_data_path':'data/GDSC1_fitted_dose_response_25Feb20.xlsx',
                 'beat_data_path':'data/beat_aml/beat_aml_aucs.csv',
                 'xenografts_data_path':'data/lung_cancer_xenografts/xenografts_value.tsv',
                 'ccle_data_path':'data/ccle/ccle_value.tsv',
                 'pancreas_data_path':'data/organoid_pancreas/organoid_value.tsv',
                 'gdsc_inchi_path':'data/gdsc_compound_inchi_smiles.csv',
                 'beat_inchi_path':'data/beat_aml/beat_aml_inhibitor_inchi_smiles.csv',
                 'xenografts_inchi_path':'data/lung_cancer_xenografts/xenografts_inhibitor_inchi_smiles.csv',
                 'ccle_inchi_path':'data/ccle/ccle_inhibitor_inchi_smiles.csv',
                 'pancreas_inchi_path':'data/organoid_pancreas/pancreas_inhibitor_inchi_smiles.csv',
                 'pancreas_gene_symbol_mapping_path':'data/organoid_pancreas/gene_name_symbol_mapping.tsv',
                 'pancreas_filename_organoid_path':'data/organoid_pancreas/file_name_to_organoid_mapping.tsv',
                 'paccmann_gene_list':'data/paccmann_gene_list.txt',
                 'gdsc_gene_list':'data/gdsc_cancer_genes.txt',
                 'oncokb_gene_list':'data/oncokb_cancer_genes.txt',
                 'lincs_gene_list':'data/lincs_cancer_genes.txt',
                 'netcore_sig_literature_mining':'data/netcore_sig_literature_mining.tab',
                 'netcore_sig_gdsc_drug_targets_literature_mining':'data/netcore_sig_gdsc_drug_targets_literature_mining.tab',
                 'netcore_sig_gdsc_drug_targets':'data/netcore_sig_gdsc_drug_targets.tab',
                 'model_param_pretrain_csv':'data/model_selection_pretrain.csv',
                 'molecule_descriptor_path':'data/molecure_descriptors.tsv'}


with open('data/params.json', 'w') as outfile:
    json.dump(param_dict, outfile)
