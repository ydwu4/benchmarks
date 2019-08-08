mpirun -np 32 -H ubuntu-4-0:8 -H ubuntu-4-2:8 -H ubuntu-4-4:8 -H ubuntu-4-6:8 \
  python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
          --model resnet101 \
          --batch_size 64 \
          --variable_update horovod
