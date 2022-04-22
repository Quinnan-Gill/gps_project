# For postgres:
```
singularity build postgres.sif docker://postgres:12.4
```

Truth == Row
Prediction == Column

Assume a true one if anytime in the window has a one


```
singularity run --env POSTGRES_PASSWORD=password -B postgres_data:/var/lib/postgresql/data -B postgres_run:/var/run/postgresql postgres.sif
```
# For running data_analysis
```
singularity build gps_research.sif docker://qisblue/gps_research
```

```
singularity run --env DATABASE_URL="postgresql://postgres:password@localhost/postgres" gps_research.sif python /home/src/data_analysis.py
```

### A Year
```
singularity run --env DATABASE_URL="postgresql://postgres:password@localhost/postgres" gps_research.sif python /home/src/data_analysis.py --start_time=2016_01_01 --end_time=2016_12_31
```

## Current status

The good stuff is on:
```
bridges2-login013
```

```
python src/bubble_prediction.py --link --start_time=2016_01_01 --end_time=2016_12_31 --message=year
```

```
python src/bubble_image.py --link --end_train_time 2016_10_01 --epochs 5
```

python src/bubble_image.py --end_train_time 2016_01_07 --epochs 5 --dataset expanded_dataset --start_val_time 2016_01_08 --end_val_time 2016_01_10

(Pdb) labels.shape
torch.Size([1564, 1, 16, 16])

```
python src/bubble_image.py --end_train_time 2016_01_20 --start_val_time 2016_01_21 --end_val_time 2016_01_30 --index_filters="1,-1" --epochs=25 --dataset expanded_dataset --experiment_name transfer_expanded --message transfer_expanded --model_path experiments/only_zeros_expanded_index_expanded_CNN.h_50/model_epoch_1.pt
```

python src/bubble_image.py --end_train_time 2016_01_02 --start_val_time 2017_01_01 --end_val_time 2017_01_20 --dataset expanded_dataset --experiment_name vae_anom --message vae_anom --model_type vae
 --model_path experiments/vae_on_zeros_index_expanded_VAE.h_50/model_epoch_20.pt --only_eval --link

 python src/bubble_image.py --end_train_time 2016_01_20 --start_val_time 2016_01_21 --end_val_time 2016_01_30 --index_filters="1,-1" --epochs=25 --dataset expanded_dataset --experiment_name filtered_lr_1e5 --learning_rate 1e-5 --message filtered_lr_1e5 --link

 python src/bubble_image.py --end_train_time 2016_01_20 --start_val_time 2016_01_21 --end_val_time 2016_01_30 --index_filters="1,-1" --dataset expanded_dataset --experiment_name filtered_batch_256 --message filtered_batch_256 --model_type vae --link --batch_size 256

 ==================================
 python src/bubble_image.py --end_train_time 2016_01_02 --start_val_time 2017_01_01 --end_val_time 2017_01_30 --dataset expanded_dataset --experiment_name eval_batch_256 --message eval_batch_256 --model_type vae --link --batch_size 256 --model_path experiments/filtered_batch_256_index_expanded_VAE.h_50/model_epoch_20.pt --only_eval --link

 python src/bubble_image.py --end_train_time 2016_01_02 --start_val_time 2017_01_01 --end_val_time 2017_01_30 --dataset expanded_dataset --experiment_name eval_batch_128 --message eval_batch_128 --model_type vae --link --batch_size 128 --model_path experiments/filtered_batch_128_index_expanded_VAE.h_50/model_epoch_19.pt --only_eval --link