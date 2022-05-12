#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 50 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 100 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 500 -train_mode drug_repurposing
#python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10000 -train_mode drug_repurposing

python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 50 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 100 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 500 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10000 -train_mode precision_oncology -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 0 -train_mode precision_oncology -flag_redo False

python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 50 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 100 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 500 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 10000 -train_mode drug_development -flag_redo False
python transfer_learning_pipeline.py -GPU 4 -source gdsc -target pancreas -use_samples 0 -train_mode drug_development -flag_redo False