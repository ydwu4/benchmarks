# Resnet101, v100(16GB) 120 <= max_batch_size <128
# Resnet101, 1080Ti(11GB) max_batch_size <=82 min_batch_size=8
# vgg16, 1080Ti(11GB) performance degradation after 48(e.g. 64), fail at 128
# enumerate batch size, num_gpu, repeating work

max_batch_size=112
min_batch_size=8
model=vgg16
step_size=8
li=(1 2 3 4 5 6 7 8)
for j in ${li[@]}; do
  for ((i=1; i<=(max_batch_size-min_batch_size)/step_size; i++)); do
    for ((k=1; k<=1; k++)); do
      per_gpu_size=$(($i*8))
      if ((per_gpu_size<=max_batch_size)) && ((per_gpu_size>=min_batch_size)); then
        echo "batch size: "$batch_size" gpu number:"$((2*j))" per gpu batch size:"$per_gpu_size
        #evenly distribute workers to two machines
        #increase to the full capacity of one worker then the other
        worker_num_0=0
        worker_num_1=0
        if ((j<=4)); then
          worker_num_0=$((2*j))
          worker_num_1=0
        else (())
          worker_num_0=8
          worker_num_1=$((2*j-8))
        fi
        echo "worker_num_0:"$worker_num_0" worker_num_1:"$worker_num_1
        mpirun -np $((2*j)) -H localhost:$worker_num_0 -H ubuntu-4-2:$worker_num_1 \
          python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
                  --model $model \
                  --batch_size $per_gpu_size \
                  --variable_update horovod
      fi
    done
  done
done

# For test range
#for ((i=2;i<=5;i++)); do
#    mpirun -np 1 -H localhost:1 \
#      python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
#              --model vgg16 \
#              --batch_size $(($i*16))\
#              --variable_update horovod
#done

# exist for legacy reason
# Evenly distribute workers across two machines
#max_batch_size=82
#min_batch_size=16
#model=resnet101
#li=(1 2 3 4 5 6 7 8)
#batch_size_list=()
#step=0
#acc_size=0
#boundary_count=0
#while ((acc_size <= 1312)); do
#  if ((acc_size>=boundary_count)); then
#    boundary_count=$((boundary_count+$max_batch_size*2))
#    step=$((step+16))
#    echo $boundary_count" step_size:"$step
#  fi
#  acc_size=$((acc_size+step))
#  batch_size_list=(${batch_size_list[@]} $acc_size)
#done;
#echo ${batch_size_list[@]}
#echo ${#batch_size_list[@]}
#
#for j in ${li[@]}; do
#  for ((i=1; i<${#batch_size_list[@]}; i++)); do
#    for ((k=1; k<=1; k++)); do
#      batch_size=${batch_size_list[$i]}
#      per_gpu_size=$(($batch_size/(j*2)))
#      if ((per_gpu_size<=max_batch_size)) && ((per_gpu_size>=min_batch_size)); then
#        echo "batch size: "$batch_size" gpu number:"$((2*j))" per gpu batch size:"$per_gpu_size
#        #evenly distribute workers to two machines
#        #mpirun -np $((2*j)) -H localhost:$j -H ubuntu-4-2:$j \
#        #increase to the full capacity of one worker then the other
#        worker_num_0=0
#        worker_num_1=0
#        if ((j<=4)); then
#          worker_num_0=$((2*j))
#          worker_num_1=0
#        else (())
#          worker_num_0=8
#          worker_num_1=$((2*j-8))
#        fi
#        echo "worker_num_0:"$worker_num_0" worker_num_1:"$worker_num_1
#        mpirun -np $((2*j)) -H localhost:$worker_num_0 -H ubuntu-4-2:$worker_num_1 \
#          python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
#                  --model $model \
#                  --batch_size $per_gpu_size \
#                  --variable_update horovod >> out_non_even_dist_resnet101 2>&1
#      fi
#    done
#  done
#done
