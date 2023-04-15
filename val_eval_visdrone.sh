DetJSON=$1

python visdrone/json_to_txt.py --out .visdrone_det_txt --gt-json data/VisDrone/annotations/val.json --det-json $DetJSON
python visdrone_eval/evaluate.py --dataset-dir data/VisDrone/annotations/val --res-dir .visdrone_det_txt
rm -rf .visdrone_det_txt

# usage:
# bash eval_visdrone.sh work_dirs/model_test/visdrone_infer.json
