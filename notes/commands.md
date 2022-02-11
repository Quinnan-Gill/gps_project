# For postgres:
```
singularity build postgres.sif docker://postgres:12.4
```


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