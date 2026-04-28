DATASET_LIST="com-youtube cit-Patents soc-LiveJournal1 com-orkut eu-2015-host arabic-2005 uk-2005 com-friendster sk-2005 twitter-2010"
# "com-youtube cit-Patents soc-LiveJournal1 com-orkut eu-2015-host arabic-2005 uk-2005 twitter-2010 com-friendster sk-2005"
# DATASET_LIST="com-friendster twitter-2010"

for data in $DATASET_LIST; do
    bash "$(dirname "$0")/run_templ_one.sh" $1 $data > time/$data.log
done
