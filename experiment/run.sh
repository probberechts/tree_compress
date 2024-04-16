DATE=$(date +"%m-%d-%Y-%H-%M")
OUTNAME="results/output_${DATE}_${HOSTNAME}"
EXPERIMENTS=$1

echo "Writing results to $OUTNAME"
echo "Running the $(cat $EXPERIMENTS | wc -l) tasks in parallel"

#python experiments.py commands $DATASETS | ./parallel --group --dryrun --slr | tee $OUTNAME


WDIR=$(pwd)

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

#python experiments.py commands $DATASETS | parallel --slf cluster --env PATH --wd $WDIR --joblog task.log --resume --progress | tee $OUTNAME

cat $EXPERIMENTS |
    parallel -j30 \
    --env OPENBLAS_NUM_THREADS \
    --env MKL_NUM_THREADS \
    --env PATH --wd $WDIR --joblog task.log --resume --progress |
    tee $OUTNAME
