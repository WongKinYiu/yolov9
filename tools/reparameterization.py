import torch
from models.yolo import Model
import argparse


def main(args):
    device = torch.device(args.device)
    model = Model(args.cfg, ch=3, nc=args.classes_num, anchors=3)
    #model = model.half()
    model = model.to(device)
    _ = model.eval()
    ckpt = torch.load(args.weights, map_location='cpu')
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if (args.model == "c" and idx < 22) or (args.model == "e" and idx < 29):
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1 if args.model == 'c' else idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif args.model == 'e' and idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx),
                               "model.{}.cv4.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx),
                               "model.{}.cv5.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx),
                               "model.{}.dfl2.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if (args.model == "c" and idx < 22) or (args.model == "e" and idx < 29):
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1 if args.model == 'c' else idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif args.model == 'e' and idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx),
                               "model.{}.cv4.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx),
                               "model.{}.cv5.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx),
                               "model.{}.dfl2.".format(idx + 16 if args.model == 'c' else idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
    _ = model.eval()

    m_ckpt = {'model': model.half(),
              'optimizer': None,
              'best_fitness': None,
              'ema': None,
              'updates': None,
              'opt': None,
              'git': None,
              'date': None,
              'epoch': -1}
    torch.save(m_ckpt, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../models/detect/gelan-c.yaml', help='model.yaml path')
    parser.add_argument('--model', type=str, default='c', help='convert model type (c or e)')
    parser.add_argument('--weights', type=str, default='./yolov9-c.pt', help='weights path')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes_num', default=80, type=int, help='number of classes')
    parser.add_argument('--save', default='./yolov9-c-converted.pt', type=str, help='save path')
    args = parser.parse_args()
    main(args)
