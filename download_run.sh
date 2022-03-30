for VARIABLE in {1..1000}
do
    python src/expanded_datasets.py --ds_start_val_time 2016_02_01 --ds_end_val_time 2016_02_29 --fetch_data
done