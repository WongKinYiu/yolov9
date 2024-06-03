# train yolov9 models
python train_dual.py --workers 8 --device 0 --batch 16 --data data/polyp_2.yaml --img 640 --cfg models/detect/yolov9-e.yaml --weights '' --name v9-e_1c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 0 --exist-ok --single-cls
python val_dual.py --data data/polyp_2.yaml --img 640 --batch 32 --conf 0.25 --iou 0.65 --device 0 --weights 'runs/v9-e_1c/weights/best.pt' --name v9-c_1c --exist-ok --single-cls
python train_dual.py --workers 8 --device 0 --batch 16 --data data/polyp_2.yaml --img 640 --cfg models/detect/yolov9-e.yaml --weights '' --name v9-e_2c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 0 --exist-ok
python val_dual.py --data data/polyp_2.yaml --img 640 --batch 32 --conf 0.25 --iou 0.65 --device 0 --weights 'runs/v9-e_2c/weights/best.pt' --name v9-e_2c --exist-ok
python train_dual.py --workers 8 --device 0 --batch 16 --data data/polyp_2.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name v9-c_1c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 300 --close-mosaic 0 --exist-ok --single-cls
python val_dual.py --data data/polyp_2.yaml --img 640 --batch 32 --conf 0.25 --iou 0.65 --device 0 --weights 'runs/v9-c_1c/weights/best.pt' --name v9-c_1c --exist-ok --single-cls
