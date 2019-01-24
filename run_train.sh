CUDA_VISIBLE_DEVICES=0 python train_cifar10.py \
	 --arch proxyless_gpu --batch-size 512 --epochs 600 \
         --learning_rate 0.025 --layers 20
