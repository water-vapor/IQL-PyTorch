NUM_EXPERIMENTS=8
for i in $(seq 1 $NUM_EXPERIMENTS)
do
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_id.pt --saveresults --saveresultspostfix id_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_para_rel.pt --saveresults --saveresultspostfix para_rel_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_perp_rel.pt --saveresults --saveresultspostfix perp_rel_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_para20.pt --saveresults --saveresultspostfix para20_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_perp20.pt --saveresults --saveresultspostfix perp20_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_para40.pt --saveresults --saveresultspostfix para40_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_perp40.pt --saveresults --saveresultspostfix perp40_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_para60.pt --saveresults --saveresultspostfix para60_$i --eval --act relu
    python evaluate.py --env antmaze-umaze-v2 --ckpt ref_ckpts_0411/relu_umaze_quantized_perp60.pt --saveresults --saveresultspostfix perp60_$i --eval --act relu

    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_id.pt --saveresults --saveresultspostfix id_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_para_rel.pt --saveresults --saveresultspostfix para_rel_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_perp_rel.pt --saveresults --saveresultspostfix perp_rel_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_para20.pt --saveresults --saveresultspostfix para20_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_perp20.pt --saveresults --saveresultspostfix perp20_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_para40.pt --saveresults --saveresultspostfix para40_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_perp40.pt --saveresults --saveresultspostfix perp40_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_para60.pt --saveresults --saveresultspostfix para60_$i --eval --act relu
    python evaluate.py --env antmaze-medium-play-v2 --ckpt ref_ckpts_0411/relu_medium_quantized_perp60.pt --saveresults --saveresultspostfix perp60_$i --eval --act relu

    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_id.pt --saveresults --saveresultspostfix id_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_para_rel.pt --saveresults --saveresultspostfix para_rel_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_perp_rel.pt --saveresults --saveresultspostfix perp_rel_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_para20.pt --saveresults --saveresultspostfix para20_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_perp20.pt --saveresults --saveresultspostfix perp20_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_para40.pt --saveresults --saveresultspostfix para40_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_perp40.pt --saveresults --saveresultspostfix perp40_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_para60.pt --saveresults --saveresultspostfix para60_$i --eval --act relu
    python evaluate.py --env antmaze-large-play-v2 --ckpt ref_ckpts_0411/relu_large_quantized_perp60.pt --saveresults --saveresultspostfix perp60_$i --eval --act relu
done
