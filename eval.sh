echo "Evaluating antmaze-umaze-v2"
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_relu/08-31-23_13.03.56_wjkn/final.pt --eval --act relu
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_relu/08-31-23_13.03.56_wjkn/final.pt --eval --multistart --act relu
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_tanh/09-01-23_15.53.51_qzfl/final.pt --eval --act tanh
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_tanh/09-01-23_15.53.51_qzfl/final.pt --eval --multistart --act tanh
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_fittedtanh/09-01-23_16.37.46_vjky/final.pt --eval --act fittedtanh
python evaluate.py --env antmaze-umaze-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-umaze-v2_fittedtanh/09-01-23_16.37.46_vjky/final.pt --eval --multistart --video --act fittedtanh

echo "Evaluating antmaze-medium-play-v2"
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_relu/09-01-23_14.58.12_schh/final.pt --eval --act relu
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_relu/09-01-23_14.58.12_schh/final.pt --eval --multistart --act relu
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_tanh/09-01-23_19.52.24_gowy/final.pt --eval --act tanh
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_tanh/09-01-23_19.52.24_gowy/final.pt --eval --multistart --act tanh
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_fittedtanh/09-01-23_17.34.35_nkcj/final.pt --eval --act fittedtanh
python evaluate.py --env antmaze-medium-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-medium-play-v2_fittedtanh/09-01-23_17.34.35_nkcj/final.pt --eval --multistart --video --act fittedtanh

echo "Evaluating antmaze-large-play-v2"
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_relu/09-01-23_21.54.50_sqgg/final.pt --eval --act relu
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_relu/09-01-23_21.54.50_sqgg/final.pt --eval --multistart --act relu
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_tanh/09-02-23_15.18.09_ylkj/final.pt --eval --act tanh
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_tanh/09-02-23_15.18.09_ylkj/final.pt --eval --multistart --act tanh
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_fittedtanh/09-01-23_23.44.04_anjo/final.pt --eval --act fittedtanh
python evaluate.py --env antmaze-large-play-v2 --ckpt /home/vapor/git/IQL-PyTorch/logs/antmaze-large-play-v2_fittedtanh/09-01-23_23.44.04_anjo/final.pt --eval --multistart --video --act fittedtanh
