# Route creater

---
Выпускная квалификационная работа, в ходе которой создавался инструмент для проектировщиков варианта дороги в первом приближении

## Требования:
* py -m pip install -r .\requirements.txt
* Python 3.8.10
## Dataset creator

- Run dataset generation:
Linux: `/bin/python3 ./dscreator.py --src-root ./data/sources/dataset/ --dst-root ./data/outputs/dataset/`
`/bin/python3 ./dscreator_new.py --src-root ./data/sources/dataset/ --dst-root ./data/outputs/dataset/`
Windows: `python.exe .\dscreator.py --src-root .\data\sources\train\ --dst-root .\data\outputs\train\`

## Model trainer

Run training:

Linux: `/bin/python3 ./trainer.py --config-file ./configs/cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`

Windows: `python.exe .\trainer.py --config-file .\configs\cfg.yaml --save-step 1 --use-tensorboard 1 --eval-step 1`

## Запуск:
* Запустите main.py: `python main.py`
* Открыть в браузере: `http://localhost:8080/`
* Подать снимок территории и карту высот для территории, где нужно проложить маршрут
* Ввести координаты начала и конца маршрута (в пикселях)
