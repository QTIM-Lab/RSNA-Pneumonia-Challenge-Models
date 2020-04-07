# Example calls for how to run the wildcat code
# this file should be on the same level as the wildcat folder containing all of the wildcat code

# --data should point to the folder that contains both images and csv files
# --model-dir should point to the folder that models should be saved to

lr=(0.1 0.01 0.001 0.0001 0.00001 0.000001)
for l in "${lr[@]}"; do
    echo 'learning rate' $l
    CUDA_VISIBLE_DEVICES=0 python3 -m wildcat.demo_chestxray --data ../ --model-dir './percent_models/' --image-size 320 --batch-size 16 --lr $l --lrp 0.1 --epochs 50 --k 1 --maps 4 --alpha 0 --adam 1 --wild 1 --dense 1
done