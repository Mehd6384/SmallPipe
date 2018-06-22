
base="/home/mehdi/Codes/ML2/NoticesNetworks/"
data=$base"DataFull/"
models=$base"/SmallerPipeline/models/"
names=small
nb_joints=3
batch_size=32
epochs=1000
mini_epochs=256
code_size=4
validation=1
lr=3e-4

nb_im=$(ls $data | grep eos | wc -l)
nb_im_f=$(($nb_im-1))


# python $base"SmallerPipeline/small_pipeline.py" --data_path $data --model_path $models \
# --names $names --code_size $code_size --nb_im $nb_im_f --nb_joints $nb_joints --validation $validation \
# --batch_size $batch_size --lr $lr --mini_epochs $mini_epochs --epochs $epochs


python $base"SmallerPipeline/test_pipeline.py" --data_path $data --model_path $models \
--names $names --code_size $code_size --nb_im $nb_im_f --nb_joints $nb_joints 