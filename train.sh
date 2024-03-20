NWAY=(5 10)
KSHOT=(1)
Q=5
train_iter=30000
val_iter=1000
test_iter=5000
val_step=1000
batch_size=4
grad_iter=1
model="multirep"

for N in ${NWAY[@]}
do
    for K in ${KSHOT[@]}
    do
        python train.py \
        --trainN $N \
        --N $N \
        --K $K \
        --Q $Q \
        --model $model \
        --batch_size $batch_size \
        --grad_iter $grad_iter \
        --train_iter $train_iter \
        --val_iter $val_iter \
        --test_iter $test_iter \
        --val_step $val_step \
        --add_loss_rdcl
    done
done