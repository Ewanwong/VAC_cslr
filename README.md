End-to-end Continuous Sign Language Recognition with Visual Alignment Constraint
Yifan Wang


How to run:
1. Put the project under the same directory as RWTH-PHOENIX-2014 dataset
    Or alternatively creature a soft link that directs to phoenix 2014 dataset under the same directory as the project
2. Resize the videos in the dataset to 224*224 by running bash ./run.sh. It takes about 15 min.
3. Download the materials and trained model from https://drive.google.com/drive/folders/1uYp0Ovi0632-UZZZg-HDbMq6TkHBxcjC?usp=sharing
    Put the downloaded folder under the project. All the data from the folder can be obtained by scripts in the project
4. Run main.ipynb to see the model performance and training process.

How to train a model:
Run python model.py --loss VAC --patience 10 --save_path model.pt --stat_path training_stat.pkl --device cuda   to train
a model with same hyperparamters as in the report

How to evaluate a model:
Run python model_evaluate.py MODEL_PATH --data test --device cuda  to evaluate a model (the performance is already given
in the report)