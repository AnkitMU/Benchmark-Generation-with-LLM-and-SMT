#!/bin/zsh
cd /Users/ankitjha/Downloads/ocl-generation-framework
PY=/opt/anaconda3/bin/python3
$PY experiments/run_parallel_suite.py --config examples/paper_ten_models_v3_vgcr.yaml --jobs 4 --logs logs/paper_v3/with_vgcr
$PY experiments/run_parallel_suite.py --config examples/paper_ten_models_v3_novgcr.yaml --jobs 4 --logs logs/paper_v3/no_vgcr
echo V3_REBUILD_DONE
