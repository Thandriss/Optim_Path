# Water segmentation

## Dataset creator

- Run dataset generation:
Linux: `/bin/python3 ./dscreator.py --src-root ./data/sources/dataset/ --dst-root ./data/outputs/dataset/`
`/bin/python3 ./dscreator_new.py --src-root ./data/sources/dataset/ --dst-root ./data/outputs/dataset/`
Windows: `python.exe .\dscreator.py --src-root .\data\sources\train\ --dst-root .\data\outputs\train\`

## Model trainer

- Run training:
Linux: `/bin/python3 ./trainer.py --config-file ./configs/cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`
Windows: `python.exe .\trainer.py --config-file .\configs\cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`

- Run tensorboard to see learning progress:
Linux: `tensorboard --logdir=./outputs/test1/tf_logs/`
Windows: `tensorboard.exe --logdir .\outputs\test\tf_logs\`

- Export:
Linux: `CUDA_VISIBLE_DEVICES=0 python3 exporter.py --config-file ./outputs/v6/cfg.yaml --batch-size 1 --target ti`
Windows: `python.exe exporter.py --config-file .\outputs\test\cfg.yaml --batch-size 1 --target ti`

## Model infer

- Run single image infer
Linux: `/bin/python3 ./inferer.py --cfg ./outputs/v5/cfg.yaml --i ./data/test/1.png --o ./data/test/results/1.png`
Windows: `python.exe .\inferer.py --cfg .\outputs\test\cfg.yaml --i .\data\infer\test.png --o .\data\infer\test_result.png`

- Run single image infer ONNX
Linux: `/bin/python3 ./inferer_onnx.py --model-path ./outputs/v6/export/onnx/imp_model.onnx --input-img ./data/test/1.png --output-img ./data/test/results/1.png --scale 0.25 --tile-size 512`

# Ships detection

## Dataset creator

- Run dataset splitting:
`/bin/python3 dscreator_ships.py --meta-path ./data/AirbusShip/metadata.json --res-dir ./data/AirbusShip/ --train-size 32768 --valid-size 8192 --min-ships 1`

## Model trainer

- Run training:
Linux: `/bin/python3 trainer_ships.py --config-file ./configs/cfg_ships.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`

- Export:
Linux: `CUDA_VISIBLE_DEVICES=0 python3 exporter_ships.py --config-file ./outputs/ships_test/cfg_ships.yaml --target ti`

## Model inferer

- Infer:
Linux: `/bin/python3 inferer_ships.py --cfg ./configs/cfg_ships.yaml --input-img ./data/ships/AistTest/main/0001_0001_30424_1_30401_01_ORT_10_02_12.png --output-img ./data/ships/AistTest/main/0001_0001_30424_1_30401_01_ORT_10_02_12_result.png --thresh 0.1`