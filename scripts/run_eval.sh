BASEDIR=~/work
if [ -z "$1" ]
  then
    NUMRUN="000"
  else
    NUMRUN=$1
fi
if [ -z "$2" ]
  then
    GAMESDIR="train"
  else
    GAMESDIR=$2
fi
if [ -z "$3" ]
  then
    NUMPROCS=1
  else
    NUMPROCS=$3
fi

export TW_RESULTS=eval_${NUMRUN}
python $BASEDIR/MagD3/scripts/ingestion_local.py --nb-processes ${NUMPROCS} \
  --in-docker $BASEDIR/MagD3/tworld_submission_gvs01 \
  $BASEDIR/0_magd3/CodaLab/${GAMESDIR}  $BASEDIR/MagD3/out/${TW_RESULTS}.json > $BASEDIR/MagD3/out/console_$NUMRUN.out
python $BASEDIR/MagD3/scripts/score.py $BASEDIR/MagD3/out/${TW_RESULTS}.json $BASEDIR/MagD3/tworld_submission_gvs01/results/ > $BASEDIR/MagD3/out/console_scoring_${TW_RESULTS}.txt
mv $BASEDIR/MagD3/tworld_submission_gvs01/results/scores.txt $BASEDIR/MagD3/tworld_submission_gvs01/results/scores_$TW_RESULTS.txt


