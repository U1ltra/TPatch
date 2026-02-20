import _init_path
import os
import math
import random
import pickle
import torch
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from PIL import Image
from classifier import *
from detector import *
from train_eval import *
from kitti import *
from tpatch import *

class AdaptiveAxes:
    def __init__(self,
                 n_figure: int,
                 n_col: int = 4,
                 fig_size: tuple = (7, 5)) -> None:
        self.n = n_figure
        self.n_col = min(n_figure, n_col)
        self.n_row = (n_figure + n_col - 1) // n_col
        self.fig_size = (fig_size[0] * self.n_col, fig_size[1] * self.n_row)
        self.fig, self.axes = plt.subplots(self.n_row,
                                           self.n_col,
                                           squeeze=False,
                                           figsize=self.fig_size)

    def __iter__(self):
        for i in range(self.n):
            j = i // self.n_col
            k = i % self.n_col
            yield self.axes[j][k]


coco_img = "dataset/mscoco/val2014"
coco_ann = "dataset/mscoco/annotations/instances_val2014.json"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0"
out_dir = "demo"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
attack_type = "HA"
target_name = "stop sign"
double_apply = False
model_type = "yolov5"
converter = LabelConverter()
if model_type == "rcnn":
    names = converter.category91
elif model_type == "yolov3":
    names = converter.category80
elif model_type == "yolov5":
    names = converter.category80
target_id = names.index(target_name)
model = get_det_model(device, model_type)

bgsize = (200, 200)

psize = (70, 170) if not double_apply else (40, 110)
relpos = (65, 15) if not double_apply else (28, 45)
relpos3 = (132, 45)

eot = True
eot_scale = 0.2
eot_angle = math.pi / 9
epoch = 50
repeat = 20
num = 100
lr = 1e-2
momentum = 0.9
beta = 3e-6 if attack_type == "CA" else 3e-6
ceta = 3e-3 if attack_type == "CA" else 3e-3
delta = 1e-6 if attack_type == "CA" else 1e-6


loader1 = load_coco(coco_img, coco_ann)
loader2 = load_kitti()

patch2 = TPatch(bgsize[0], bgsize[1], target_id, device, eot=True, 
                eot_scale=eot_scale, eot_angle=eot_angle, p=1)
resize = tv.transforms.Resize(bgsize)
quick_load = lambda x: resize(patch2.pil2tensor(Image.open(x))).unsqueeze(0).to(device)

patch2.data = quick_load("stop_sign.png")
patch2.load_mask("stop_sign_mask.png")
patch2.rotate_mask = resize(patch2.rotate_mask)

if attack_type == "CA":
    content = "pikachu.jpg"
elif attack_type == "HA":
    if not double_apply:
        content = "stop.png"
    else:
        content = "usenix_text.png"

a = tv.models.vgg19(False).to(device)
a.load_state_dict(torch.load("weights/vgg19-dcbb9e9d.pth"))
content_loss = ContentLoss(a.features, content, device, extract_layer=11)
tv_loss = TVLoss()

filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"

# === eval only ===
# filename = "137498.png"
# === eval only ===

if attack_type == "HA":
    patch = TPatch(psize[0], psize[1], target_id, device=device, lr=lr, momentum=momentum, 
                   eot=eot, eot_scale=0.97, eot_angle=math.pi/60)
else:
    patch = TPatch(bgsize[0], bgsize[1], target_id, device=device, lr=lr, momentum=momentum, 
                   eot=eot, eot_scale=eot_scale, eot_angle=eot_angle, p=1)
save_path = os.path.join(out_dir, filename)

if os.path.exists(os.path.join(out_dir, filename)):
    patch.load(save_path)

logfile = (save_path).replace("png", "txt")
print("=" * 40)
print(datetime.now())
print(save_path)
print(attack_type)
print(model_type)
print(beta, ceta, delta, psize)


def train(train_loader):
    if model_type.startswith("rcnn"):
        model.train()
    else:
        model.eval()
    t1 = datetime.now()
    log_loss = torch.zeros(3, device=device)
    
    pbar = tqdm(total=epoch * repeat)
    pbar.set_description("Training")
    pbar.update(0)
    
    for i, img in enumerate(train_loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        for j in range(repeat):
            if attack_type == "CA":
                pos = patch.random_pos((h, w))
                imgo = patch.apply(img, pos)
                gt_box, _ = _make_boxes(patch, pos, model_type[:4].upper())
                last_scale = patch.last_scale
            elif attack_type == "HA":
                pos2 = patch2.random_pos((h, w))
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                relpos2 = (relpos[0] + dx, relpos[1] + dy)
                patch2.data = patch.apply(quick_load("stop_sign.png"), relpos2, do_random_color=True)
                if double_apply:
                    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                    relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                    patch2.data = patch.apply(patch2.data, relpos2, do_random_color=True)
                imgo = patch2.apply(img, pos2, do_random_color=False)
                gt_box, _ = _make_boxes(patch2, pos2, model_type[:4].upper())
                last_scale = patch2.last_scale

            loss1 = model(imgo, gt_box, hiding=True)
            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss = (1/last_scale**2)*loss1 + beta*loss3 + ceta*loss4
            if torch.isnan(loss).any(): continue
            log_loss += torch.tensor((loss1.item(), loss3.item(), loss4.item()), device=device)
            patch.update(loss)

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "loss1": f"{loss1.item():.4f}",
                "loss3": f"{loss3.item():.4f}",
                "loss4": f"{loss4.item():.4f}"
            })
            
        if i == 1:
            t2 = datetime.now()
            pred_time = (t2-t1) * (epoch-1)
            print("pred time:", pred_time)
        if i == epoch:
            break
    return log_loss / (epoch * repeat)


def eval(test_loader, results_path=None):
    model.eval()
    t1 = datetime.now()
    success = 0
    results = []

    pbar = tqdm(total=num)
    pbar.set_description("Evaluating")
    pbar.update(0)

    for i, img in enumerate(test_loader, 1):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        set_resize = random.uniform(eot_scale, 1)
        set_rotate = random.uniform(-eot_angle, eot_angle)

        if attack_type == "CA":
            pos = patch.random_pos((h, w))
            imgo = patch.apply(img, pos, test_mode=True, set_resize=set_resize,
                               set_rotate=set_rotate)
        elif attack_type == "HA":
            pos = patch2.random_pos((h, w))
            relpos2 = (relpos[0], relpos[1])
            patch2.data = patch.apply(quick_load("stop_sign.png"), relpos2, test_mode=True, do_random_color=False)
            if double_apply:
                relpos2 = (relpos3[0], relpos3[1])
                patch2.data = patch.apply(patch2.data, relpos2, test_mode=True, do_random_color=False)
            imgo = patch2.apply(img, pos, test_mode=True, set_resize=set_resize, set_rotate=set_rotate, do_random_color=True)

        # get both post-NMS detections and the raw pre-NMS tensor
        nms_preds, pred_raw = model(imgo, return_raw=True)
        pred = nms_preds[0]   # post-NMS: [K, 6] = [x1,y1,x2,y2,conf,cls]

        # --- pre-NMS: filter to objectness > 0.01 to keep file size manageable ---
        raw = pred_raw[0]                        # [N, 5+C]
        keep = raw[:, 4] > 0.01
        raw_kept = raw[keep].cpu()
        raw_boxes = raw_kept[:, :4].numpy()      # xywh (center format, pixel coords)
        raw_obj   = raw_kept[:, 4].numpy()       # objectness score
        raw_cls   = raw_kept[:, 5:].numpy()      # per-class probabilities [M, 80]

        if attack_type == "CA":
            pw, ph = patch.w, patch.h
        elif attack_type == "HA":
            pw, ph = patch2.w, patch2.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize) * pw * 0.5,
            pos[0] + (1 - set_resize) * ph * 0.5,
            pos[1] + (1 + set_resize) * pw * 0.5,
            pos[0] + (1 + set_resize) * ph * 0.5,
            patch.target,
        ]])

        flag = isappear(pred.cpu(), gt_box)

        s = 0
        if attack_type == "HA":
            if not flag:
                s = 1
        elif attack_type == "CA":
            if flag:
                s = 1
        success += s

        results.append({
            "image_idx":   i,
            "set_resize":  set_resize,
            "set_rotate":  np.degrees(set_rotate),
            "patch_pos":   pos,
            "gt_box":      gt_box.numpy(),
            # post-NMS final detections: [x1,y1,x2,y2,conf,cls]
            "nms_detections": pred.cpu().numpy(),
            # pre-NMS (obj > 0.01): raw boxes/scores before any filtering
            "raw_boxes":   raw_boxes,   # [M,4] xywh center-format pixel coords
            "raw_obj":     raw_obj,     # [M]   objectness
            "raw_cls":     raw_cls,     # [M,80] per-class probability
            "attack_success": s,
        })

        pbar.update(1)

        if i == 10:
            t2 = datetime.now()
            pred_time = (t2 - t1) * (num - 10) / 10
            print("pred time:", pred_time)
        if i == num:
            break

    if results_path is not None:
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_path}")

    return success, results


eval_only = False  # set True to skip training and only run evaluation

def main():
    results_path = save_path.replace(".png", "_results.pkl")
    decay_epoch = 2
    n_decay = 3
    for e in range(1, decay_epoch * n_decay + 1):
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        if not eval_only:
            print(f"Epoch {e}: start training...")
            losses = train(loader1)
            print(losses)

        print(f"Epoch {e}: start evaluating...")
        with torch.no_grad():
            sucs, _ = eval(loader2, results_path=results_path)
        print(f"Attack success: {sucs}/{num}")

        if not eval_only:
            patch.save(save_path)
            if e % decay_epoch == 0:
                patch.opt.lr *= 0.3

        if eval_only:
            break  # single eval pass when not training

if __name__ == "__main__":
    main()
