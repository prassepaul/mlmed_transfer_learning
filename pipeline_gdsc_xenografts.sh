#python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 10 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 50 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 10000 -train_mode drug_repurposing

python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 10 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 50 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 1 -source gdsc -target xenografts -use_samples 10000 -train_mode precision_oncology -flag_redo False

#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target xenografts -use_samples 10 -train_mode drug_development -n_splits 3
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target xenografts -use_samples 50 -train_mode drug_development -n_splits 3
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target xenografts -use_samples 10000 -train_mode drug_development -n_splits 3
