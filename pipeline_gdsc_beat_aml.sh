#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 10 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 50 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 100 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 500 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 1000 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 5000 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 10000 -train_mode drug_repurposing
#
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 10 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 50 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 100 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 500 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 1000 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 5000 -train_mode precision_oncology -flag_redo False
#python transfer_learning_pipeline.py -GPU 0 -source gdsc -target beat_aml -use_samples 10000 -train_mode precision_oncology -flag_redo False
#
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 10 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 50 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 100 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 500 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 1000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 5000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 7 -source gdsc -target beat_aml -use_samples 10000 -train_mode drug_development -flag_redo False

#python transfer_learning_pipeline.py -GPU 2 -source gdsc -target beat_aml -use_samples 10000 -train_mode precision_oncology -save_dir results_tmp/
