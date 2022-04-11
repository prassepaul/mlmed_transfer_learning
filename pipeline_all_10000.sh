python transfer_learning_pipeline.py -GPU 4 -source gdsc -target beat_aml -use_samples 10000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target beat_aml -use_samples 10000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target xenografts -use_samples 10000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target xenografts -use_samples 10000 -train_mode drug_development -n_splits 3 -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target ccle -use_samples 10000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target ccle -use_samples 10000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target beat_aml -use_samples 20000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target beat_aml -use_samples 20000 -train_mode drug_development -flag_redo False
