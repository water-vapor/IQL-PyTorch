for i in $(seq 1 $NUM_EXPERIMENTS)
do
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze.pt --saveresults --saveresultspostfix cid_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium.pt --saveresults --saveresultspostfix cid_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large.pt --saveresults --saveresultspostfix cid_$i --eval --act relu
done
