
data_path="/home/mehdi/Codes/ML2/NoticesNetworks/DataFull/"
model_path="/home/mehdi/Codes/ML2/NoticesNetworks/FullPipeline/models/"
code_size=8
joints=3
frames=60000
model_name=Full

# ----------------------------
## FIRST: Generating the DATA
# ----------------------------

python /home/mehdi/Codes/ML2/NoticesNetworks/Environment/DataCreation.py --frames $frames --obs 0 \
--path $data_path --frameskip 5 --dim 64 --save_deltas True --ask_validation 1 --joints $joints --use_eos 1 --target_limits 0.1 0.9 0.1 0.9 \
--max_steps 500 

echo "Data creation done"
nb_im=$(ls $data_path | grep im | wc -l) # Get number of datapoints
nb_im_future=$(($nb_im-1))

# ----------------------------
# TRAIN first perception module
# ----------------------------

# echo "Starting perception module training"

# python /home/mehdi/Codes/ML2/NoticesNetworks/Perception/ae_train.py --data_path $data_path --data_output $model_path --model_name $model_name \
# --nb_im $nb_im --mini_epochs 200 --epochs 100 --scheduler_step_size 100 --scheduler_gamma 0.9 --lr 5e-5 --batch_size 32 --code_size $code_size \
# --weight_decay 0 --ask_validation 0 --use_display 0 --load 1 --load_same 1 --use_cuda 1 

# echo "Perception module training done"

# # ----------------------------
# # Output autoencoder representation to train next modules
# # ----------------------------

# python /home/mehdi/Codes/ML2/NoticesNetworks/Perception/ae_output.py --data_path $data_path --data_output $model_path --model_name $model_name \
# --nb_im $nb_im --code_size $code_size --ask_validation 0 

# # echo "Perception module Z output done"

# # ----------------------------
# # Train Task to generate next state given current state + target location 
# # ----------------------------

# echo "Starting training task module "
# python /home/mehdi/Codes/ML2/NoticesNetworks/Tasks/train_task.py --data_path $data_path --data_output $model_path --model_name $model_name \
# --nb_im $nb_im_future --code_size $code_size --batch_size 32 --mini_epochs 200 --epochs 500 --scheduler_step_size 550 \
# --scheduler_gamma 0.7 --lr 3e-4 --ask_validation 0 --use_display 0 --load 0 --load_same 1

# echo "Task module training done"

# # ----------------------------
# # Output task representation to train control module
# # ----------------------------

# python /home/mehdi/Codes/ML2/NoticesNetworks/Tasks/task_output.py --data_path $data_path --data_output $model_path --model_name $model_name \
# --nb_im $nb_im_future --code_size $code_size --ask_validation 0 

# echo "Task module P output done"

# # ----------------------------
# # Finally, train control module 
# # ----------------------------

# python /home/mehdi/Codes/ML2/NoticesNetworks/Control/train_control.py --data_path $data_path --data_output $model_path --model_name $model_name \
# --nb_im $(($nb_im_future-1)) --output_size $joints --mini_epochs 150 --epochs 1000 --scheduler_step_size 200 --scheduler_gamma 0.7 --lr 3e-4 --batch_size 128 --code_size $code_size \
# --ask_validation 0 --use_display 0 --load 0 --load_same 1

# echo "Training complete. Launching project skynet"
