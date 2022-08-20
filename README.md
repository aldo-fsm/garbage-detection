# Projeto Final - Modelos Preditivos Conexionistas

### Aldo Ferreira de Souza Monteiro

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
|Deteção de Objetos|YOLOv5|PyTorch|

## Performance
  - mAP_0.5 = 0.18922
  - mAP_0.5:0.95 = 0.08857
  - precision = 0.21019
  - recall = 0.25713

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
    wandb: Currently logged in as: afsm (yolo-garbage-detection). Use `wandb login --relogin` to force relogin
train: weights=yolov5s.pt, cfg=, data=/content/yolov5/Garbage-detection-1/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=4000, batch_size=64, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-10-g5c854fa Python-3.7.13 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.13.1
wandb: Run data is saved locally in /content/yolov5/wandb/run-20220818_023407-1wgcbwdz
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run neat-elevator-1
wandb: ⭐️ View project at https://wandb.ai/yolo-garbage-detection/YOLOv5
wandb: 🚀 View run at https://wandb.ai/yolo-garbage-detection/YOLOv5/runs/1wgcbwdz
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 142MB/s]
YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected.
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt to yolov5s.pt...
100% 14.1M/14.1M [00:00<00:00, 291MB/s]

Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 270 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/yolov5/Garbage-detection-1/train/labels' images and labels...153 found, 0 missing, 0 empty, 0 corrupt: 100% 153/153 [00:00<00:00, 2177.95it/s]
train: New cache created: /content/yolov5/Garbage-detection-1/train/labels.cache
train: Caching images (0.2GB ram): 100% 153/153 [00:01<00:00, 146.34it/s]
val: Scanning '/content/yolov5/Garbage-detection-1/valid/labels' images and labels...14 found, 0 missing, 0 empty, 0 corrupt: 100% 14/14 [00:00<00:00, 478.13it/s]
val: New cache created: /content/yolov5/Garbage-detection-1/valid/labels.cache
val: Caching images (0.0GB ram): 100% 14/14 [00:00<00:00, 67.65it/s]
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 5.47 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 4000 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    0/3999     14.2G    0.1169    0.1121   0.04101       618       640: 100% 3/3 [00:05<00:00,  1.81s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:02<00:00,  2.04s/it]
                 all         14        150    0.00602      0.143    0.00573    0.00178

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    1/3999     14.2G    0.1146    0.1101   0.04009       555       640: 100% 3/3 [00:01<00:00,  1.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.07it/s]
                 all         14        150    0.00693      0.168    0.00556     0.0017

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    2/3999     14.2G     0.111    0.1063   0.03972       334       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.45it/s]
                 all         14        150    0.00772      0.191    0.00619    0.00155

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    3/3999     14.2G    0.1059    0.1232   0.03867       430       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.38it/s]
                 all         14        150     0.0083      0.216    0.00956    0.00209

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    4/3999     14.2G    0.1008    0.1158    0.0379       404       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.08it/s]
                 all         14        150     0.0214     0.0732     0.0171    0.00445

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    5/3999     14.2G   0.09783      0.13   0.03568       402       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.48it/s]
                 all         14        150     0.0298      0.048     0.0225    0.00662

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    6/3999     14.2G   0.09282    0.1364   0.03486       442       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0% 0/1 [00:00<?, ?it/s]WARNING: NMS time limit 0.720s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.07it/s]
                 all         14        150     0.0257      0.106     0.0201    0.00536

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    7/3999     14.2G   0.08965    0.1298   0.03311       471       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0% 0/1 [00:00<?, ?it/s]WARNING: NMS time limit 0.720s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:01<00:00,  1.05s/it]
                 all         14        150      0.373      0.107     0.0301    0.00738

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    8/3999     14.2G   0.08692    0.1395   0.03291       425       640: 100% 3/3 [00:01<00:00,  1.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0% 0/1 [00:00<?, ?it/s]WARNING: NMS time limit 0.720s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.12it/s]
                 all         14        150      0.391      0.128     0.0414      0.012

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    9/3999     14.2G   0.08209    0.1348   0.03118       434       640: 100% 3/3 [00:01<00:00,  1.69it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.28it/s]
                 all         14        150      0.395      0.148      0.044     0.0131

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   10/3999     14.2G   0.07952    0.1252   0.03138       418       640: 100% 3/3 [00:01<00:00,  1.66it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.11it/s]
                 all         14        150      0.389      0.186     0.0524     0.0131

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   11/3999     14.2G   0.07758    0.1285   0.03086       354       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0% 0/1 [00:00<?, ?it/s]WARNING: NMS time limit 0.720s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.05it/s]
                 all         14        150      0.387      0.167     0.0589     0.0177

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   12/3999     14.2G   0.07588    0.1198   0.02941       415       640: 100% 3/3 [00:01<00:00,  1.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         14        150       0.39      0.216     0.0618     0.0184

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   13/3999     14.2G   0.07167    0.1163    0.0293       323       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.35it/s]
                 all         14        150      0.403      0.188     0.0643     0.0185

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   14/3999     14.2G   0.07257    0.1392   0.02853       468       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.65it/s]
                 all         14        150      0.723      0.132     0.0555     0.0154

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   15/3999     14.2G   0.07201    0.1361   0.02727       429       640: 100% 3/3 [00:01<00:00,  1.64it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.11it/s]
                 all         14        150      0.381      0.213     0.0539     0.0157

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   16/3999     14.2G   0.07114    0.1271   0.02752       430       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0% 0/1 [00:00<?, ?it/s]WARNING: NMS time limit 0.720s exceeded
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:01<00:00,  1.08s/it]
                 all         14        150      0.396      0.163     0.0832     0.0247

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   17/3999     14.2G   0.06937    0.1251   0.02725       390       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.39it/s]
                 all         14        150      0.457      0.103     0.0772     0.0214

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   18/3999     14.2G    0.0726    0.1292   0.02649       474       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.64it/s]
                 all         14        150      0.408       0.27     0.0898     0.0258

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   19/3999     14.2G   0.06935    0.1266   0.02551       490       640: 100% 3/3 [00:01<00:00,  1.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.83it/s]
                 all         14        150      0.413      0.246     0.0976     0.0308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   20/3999     14.2G   0.06715    0.1182   0.02484       447       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.68it/s]
                 all         14        150      0.422      0.242      0.107     0.0314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   21/3999     14.2G   0.07156    0.1212   0.02406       420       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.79it/s]
                 all         14        150      0.454      0.218      0.116     0.0348

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   22/3999     14.2G    0.0689    0.1196   0.02429       400       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.64it/s]
                 all         14        150      0.456      0.213      0.104     0.0312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   23/3999     14.2G    0.0687    0.1185   0.02181       459       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.51it/s]
                 all         14        150      0.437      0.252      0.137     0.0448

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   24/3999     14.2G   0.06807    0.1367   0.02251       619       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.64it/s]
                 all         14        150       0.55      0.148      0.152     0.0486

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   25/3999     14.2G   0.06647    0.1207   0.02252       349       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.35it/s]
                 all         14        150      0.433      0.246       0.11     0.0303

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   26/3999     14.2G   0.06657    0.1225   0.02081       436       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.11it/s]
                 all         14        150      0.477      0.183      0.111     0.0401

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   27/3999     14.2G    0.0642    0.1199    0.0194       422       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.31it/s]
                 all         14        150      0.519      0.115      0.111     0.0366

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   28/3999     14.2G   0.06453    0.1197   0.01817       454       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.91it/s]
                 all         14        150      0.505      0.115      0.105     0.0368

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   29/3999     14.2G    0.0662    0.1176   0.01843       419       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.67it/s]
                 all         14        150      0.506      0.103       0.11     0.0396

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   30/3999     14.2G   0.06299    0.1215   0.01801       490       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.79it/s]
                 all         14        150      0.426      0.196     0.0948     0.0289

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   31/3999     14.2G   0.06488    0.1132   0.01649       441       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.16it/s]
                 all         14        150      0.481      0.191      0.127     0.0463

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   32/3999     14.2G   0.06417    0.1187   0.01608       437       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.94it/s]
                 all         14        150      0.561      0.164      0.169     0.0602

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   33/3999     14.2G   0.06325    0.1262     0.017       507       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.34it/s]
                 all         14        150      0.598      0.145      0.129     0.0486

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   34/3999     14.2G   0.05969    0.1134   0.01557       417       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.22it/s]
                 all         14        150      0.516      0.184      0.129      0.045

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   35/3999     14.2G    0.0631    0.1179   0.01425       567       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.52it/s]
                 all         14        150      0.474      0.218      0.121     0.0458

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   36/3999     14.2G    0.0604    0.1116   0.01504       486       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.38it/s]
                 all         14        150      0.557      0.137       0.11     0.0407

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   37/3999     14.2G   0.05891    0.1163   0.01573       418       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.37it/s]
                 all         14        150      0.514      0.185      0.132     0.0474

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   38/3999     14.2G    0.0606     0.115   0.01459       551       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.57it/s]
                 all         14        150      0.472       0.22      0.108     0.0343

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   39/3999     14.2G   0.05851     0.117   0.01508       393       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.09it/s]
                 all         14        150      0.457      0.149      0.086     0.0271

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   40/3999     14.2G   0.05724    0.1056    0.0147       412       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.60it/s]
                 all         14        150      0.449      0.141     0.0577     0.0207

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   41/3999     14.2G    0.0557    0.1131   0.01392       485       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.96it/s]
                 all         14        150      0.408     0.0951     0.0447     0.0166

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   42/3999     14.2G   0.05747    0.1027   0.01395       418       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.59it/s]
                 all         14        150       0.45      0.127     0.0581     0.0205

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   43/3999     14.2G   0.05378    0.1111    0.0139       495       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.53it/s]
                 all         14        150      0.492      0.171      0.105     0.0344

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   44/3999     14.2G   0.05449   0.09748   0.01402       416       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.20it/s]
                 all         14        150     0.0724      0.116     0.0482     0.0193

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   45/3999     14.2G   0.05264    0.1016   0.01271       418       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.28it/s]
                 all         14        150      0.481      0.133     0.0862     0.0263

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   46/3999     14.2G   0.05603    0.1103   0.01207       581       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.06it/s]
                 all         14        150      0.498      0.217      0.132     0.0494

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   47/3999     14.2G   0.05087    0.1024   0.01189       458       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.38it/s]
                 all         14        150      0.123      0.164     0.0762     0.0252

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   48/3999     14.2G    0.0549    0.1034   0.01262       487       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.37it/s]
                 all         14        150      0.463      0.204      0.115      0.042

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   49/3999     14.2G   0.05279   0.09326   0.01185       344       640: 100% 3/3 [00:02<00:00,  1.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.33it/s]
                 all         14        150      0.471      0.182      0.096     0.0313

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   50/3999     14.2G   0.04987    0.1174   0.01182       621       640: 100% 3/3 [00:03<00:00,  1.04s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.13it/s]
                 all         14        150      0.527      0.149      0.138      0.053

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   51/3999     14.2G   0.05279    0.1074   0.01216       503       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.64it/s]
                 all         14        150      0.561      0.163      0.135     0.0406

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   52/3999     14.2G   0.05214   0.09362   0.01151       380       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.10it/s]
                 all         14        150      0.524      0.206       0.14     0.0473

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   53/3999     14.2G   0.04996    0.1083  0.009334       491       640: 100% 3/3 [00:02<00:00,  1.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.12it/s]
                 all         14        150      0.549      0.125      0.121     0.0522

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   54/3999     14.2G   0.05072    0.0997   0.01036       489       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.23it/s]
                 all         14        150       0.49      0.215      0.159     0.0673

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   55/3999     14.2G   0.05197    0.1041  0.009802       476       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.42it/s]
                 all         14        150      0.526      0.188      0.142     0.0476

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   56/3999     14.2G   0.05184   0.09859  0.009772       450       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.34it/s]
                 all         14        150      0.495      0.181      0.126     0.0426

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   57/3999     14.2G   0.05038    0.1002  0.009713       466       640: 100% 3/3 [00:01<00:00,  1.63it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.97it/s]
                 all         14        150      0.506      0.156      0.112     0.0426

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   58/3999     14.2G   0.04825   0.09343   0.00955       447       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.08it/s]
                 all         14        150        0.5      0.172      0.141       0.05

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   59/3999     14.2G   0.05446     0.101   0.00877       624       640: 100% 3/3 [00:03<00:00,  1.13s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.83it/s]
                 all         14        150      0.576      0.176      0.134     0.0553

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   60/3999     14.2G   0.04881    0.1011  0.008902       575       640: 100% 3/3 [00:02<00:00,  1.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.56it/s]
                 all         14        150      0.518      0.137      0.121      0.048

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   61/3999     14.2G   0.05144   0.09497  0.008915       442       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.17it/s]
                 all         14        150      0.487      0.168      0.118     0.0457

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   62/3999     14.2G   0.04594    0.0948  0.008645       402       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.40it/s]
                 all         14        150      0.489      0.188      0.139     0.0575

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   63/3999     14.2G   0.04958   0.09522  0.008157       457       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.00it/s]
                 all         14        150      0.514      0.227      0.167     0.0639

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   64/3999     14.2G    0.0481   0.08948  0.007923       401       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.55it/s]
                 all         14        150      0.519      0.247      0.172     0.0783

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   65/3999     14.2G   0.04993   0.09849  0.007174       563       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.11it/s]
                 all         14        150      0.543      0.172      0.174     0.0729

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   66/3999     14.2G   0.04781   0.08265  0.007366       402       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.60it/s]
                 all         14        150      0.508      0.257      0.167     0.0642

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   67/3999     14.2G   0.04681   0.09199  0.007861       542       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.40it/s]
                 all         14        150      0.186      0.239      0.145     0.0485

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   68/3999     14.2G   0.04699   0.09671  0.007075       500       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.62it/s]
                 all         14        150      0.192      0.243      0.141     0.0516

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   69/3999     14.2G   0.04927   0.08354  0.007944       416       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.78it/s]
                 all         14        150      0.176      0.242      0.133      0.044

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   70/3999     14.2G   0.04542   0.09348  0.007333       359       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.80it/s]
                 all         14        150      0.153      0.278      0.132     0.0528

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   71/3999     14.2G   0.04522   0.09402  0.006783       518       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.26it/s]
                 all         14        150       0.13      0.241      0.129     0.0457

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   72/3999     14.2G   0.04596   0.09872  0.006791       501       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.28it/s]
                 all         14        150       0.15       0.22      0.138     0.0582

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   73/3999     14.2G   0.04776   0.07651  0.007187       311       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.00it/s]
                 all         14        150      0.209      0.163      0.127     0.0475

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   74/3999     14.2G   0.04554   0.09909   0.00685       646       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.75it/s]
                 all         14        150      0.546      0.181      0.148     0.0584

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   75/3999     14.2G   0.04841   0.08699  0.006649       438       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.04it/s]
                 all         14        150      0.174       0.27      0.159     0.0603

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   76/3999     14.2G   0.04484   0.09049  0.006333       525       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.41it/s]
                 all         14        150        0.6        0.2      0.177     0.0668

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   77/3999     14.2G   0.04552   0.09417  0.005451       496       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.13it/s]
                 all         14        150      0.564      0.228      0.168     0.0672

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   78/3999     14.2G   0.04712   0.09029  0.005336       499       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.78it/s]
                 all         14        150      0.565      0.263      0.169     0.0707

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   79/3999     14.2G   0.04484    0.1074  0.005244       721       640: 100% 3/3 [00:02<00:00,  1.17it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.51it/s]
                 all         14        150      0.507      0.286      0.142     0.0549

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   80/3999     14.2G   0.04627   0.09075  0.005397       414       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.45it/s]
                 all         14        150      0.184      0.294      0.177     0.0654

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   81/3999     14.2G   0.04352   0.09637  0.005387       569       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.22it/s]
                 all         14        150      0.225      0.254      0.191     0.0716

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   82/3999     14.2G   0.04365   0.08538  0.005007       473       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.22it/s]
                 all         14        150      0.218      0.245      0.185     0.0725

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   83/3999     14.2G   0.04271    0.0852  0.005873       451       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.87it/s]
                 all         14        150       0.57      0.243      0.188       0.07

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   84/3999     14.2G   0.04624    0.1008  0.005229       573       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.23it/s]
                 all         14        150        0.5       0.26      0.165     0.0641

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   85/3999     14.2G   0.04431   0.09222  0.005318       515       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.18it/s]
                 all         14        150      0.202       0.21       0.17     0.0614

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   86/3999     14.2G   0.04502   0.08307  0.004583       457       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.19it/s]
                 all         14        150       0.51      0.181      0.134      0.048

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   87/3999     14.2G   0.04253   0.09443  0.004874       576       640: 100% 3/3 [00:02<00:00,  1.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.33it/s]
                 all         14        150      0.153      0.207      0.141     0.0557

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   88/3999     14.2G   0.04169   0.08624  0.004714       473       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.13it/s]
                 all         14        150      0.194      0.156      0.146     0.0545

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   89/3999     14.2G   0.04258   0.08592  0.004524       457       640: 100% 3/3 [00:02<00:00,  1.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.86it/s]
                 all         14        150      0.185      0.171      0.138     0.0437

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   90/3999     14.2G   0.04259   0.08683  0.004432       524       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.79it/s]
                 all         14        150      0.196      0.177      0.113       0.04

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   91/3999     14.2G   0.04383    0.0825  0.004545       423       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.72it/s]
                 all         14        150      0.173      0.163      0.109     0.0349

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   92/3999     14.2G   0.04042   0.08262  0.004891       393       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.05it/s]
                 all         14        150      0.215      0.161      0.134     0.0433

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   93/3999     14.2G   0.04566   0.09086   0.00429       580       640: 100% 3/3 [00:02<00:00,  1.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.97it/s]
                 all         14        150      0.202       0.15       0.12     0.0437

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   94/3999     14.2G   0.04198   0.08053  0.004738       480       640: 100% 3/3 [00:02<00:00,  1.20it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.15it/s]
                 all         14        150      0.204       0.18      0.144     0.0523

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   95/3999     14.2G   0.04107   0.08194  0.004499       315       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.67it/s]
                 all         14        150      0.161      0.177      0.102     0.0314

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   96/3999     14.2G   0.04484   0.08216  0.004298       397       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.73it/s]
                 all         14        150      0.158      0.216      0.102     0.0341

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   97/3999     14.2G   0.04142   0.08754  0.004593       526       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.76it/s]
                 all         14        150      0.178      0.239      0.134     0.0434

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   98/3999     14.2G   0.04227   0.08635  0.004575       565       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.31it/s]
                 all         14        150      0.149      0.258      0.137     0.0521

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   99/3999     14.2G    0.0421   0.08389   0.00476       462       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.57it/s]
                 all         14        150      0.241      0.212      0.135     0.0508

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  100/3999     14.2G   0.04102   0.09795  0.004382       645       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.28it/s]
                 all         14        150       0.56      0.181      0.167     0.0657

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  101/3999     14.2G   0.04231   0.08157   0.00404       427       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.77it/s]
                 all         14        150      0.554      0.191      0.155     0.0582

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  102/3999     14.2G   0.03919   0.08692  0.004351       352       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.00it/s]
                 all         14        150      0.257      0.301       0.16      0.054

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  103/3999     14.2G   0.04105   0.08165  0.003966       429       640: 100% 3/3 [00:02<00:00,  1.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.71it/s]
                 all         14        150      0.301      0.265       0.17     0.0628

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  104/3999     14.2G   0.04075   0.08616   0.00379       409       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.86it/s]
                 all         14        150      0.278      0.305       0.17     0.0569

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  105/3999     14.2G   0.04116   0.07639   0.00443       368       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.04it/s]
                 all         14        150      0.153      0.232      0.126     0.0477

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  106/3999     14.2G    0.0384   0.08006  0.004358       450       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.84it/s]
                 all         14        150      0.163      0.247      0.128     0.0512

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  107/3999     14.2G   0.04229   0.08411  0.003796       537       640: 100% 3/3 [00:02<00:00,  1.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.10it/s]
                 all         14        150      0.184      0.198      0.129     0.0507

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  108/3999     14.2G   0.03806   0.08124  0.003664       488       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.10it/s]
                 all         14        150       0.21      0.181      0.137     0.0553

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  109/3999     14.2G   0.04132    0.0821  0.004036       507       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.67it/s]
                 all         14        150      0.205      0.214      0.148     0.0583

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  110/3999     14.2G   0.03992    0.0797  0.004074       404       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.14it/s]
                 all         14        150      0.217      0.186      0.146     0.0614

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  111/3999     14.2G   0.03816   0.07983  0.003943       475       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.92it/s]
                 all         14        150      0.202       0.23      0.153     0.0628

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  112/3999     14.2G   0.03985   0.08621  0.004211       478       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.05it/s]
                 all         14        150      0.198      0.236      0.156     0.0568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  113/3999     14.2G   0.03991   0.07578  0.003604       411       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.14it/s]
                 all         14        150      0.208        0.2      0.147     0.0481

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  114/3999     14.2G   0.03923   0.07551  0.004079       410       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.09it/s]
                 all         14        150      0.542       0.18       0.15     0.0599

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  115/3999     14.2G   0.04071    0.0767  0.003509       494       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.48it/s]
                 all         14        150      0.546      0.196      0.146     0.0529

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  116/3999     14.2G   0.03953   0.08441  0.003591       522       640: 100% 3/3 [00:02<00:00,  1.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.28it/s]
                 all         14        150      0.599      0.197      0.174     0.0661

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  117/3999     14.2G   0.03824   0.08068  0.003724       464       640: 100% 3/3 [00:02<00:00,  1.14it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.04it/s]
                 all         14        150      0.545      0.215      0.156     0.0614

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  118/3999     14.2G   0.03947   0.07499  0.003672       508       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.12it/s]
                 all         14        150      0.575      0.205      0.182     0.0687

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  119/3999     14.2G   0.03774   0.08151   0.00352       454       640: 100% 3/3 [00:02<00:00,  1.07it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.88it/s]
                 all         14        150      0.221      0.198      0.157     0.0568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  120/3999     14.2G   0.04044   0.07295  0.003923       394       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.23it/s]
                 all         14        150      0.552      0.171      0.139     0.0528

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  121/3999     14.2G   0.03852   0.07739  0.003534       456       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.35it/s]
                 all         14        150      0.547      0.226      0.168     0.0603

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  122/3999     14.2G    0.0379   0.07377  0.003391       363       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.46it/s]
                 all         14        150        0.2      0.242      0.171     0.0582

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  123/3999     14.2G   0.03824   0.08245  0.003888       533       640: 100% 3/3 [00:01<00:00,  1.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.44it/s]
                 all         14        150      0.185      0.222       0.13     0.0431

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  124/3999     14.2G   0.03794   0.07784  0.003404       496       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.01it/s]
                 all         14        150      0.173      0.253       0.15     0.0533

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  125/3999     14.2G   0.03889   0.07451  0.003991       360       640: 100% 3/3 [00:02<00:00,  1.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.78it/s]
                 all         14        150      0.157      0.218       0.14     0.0507

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  126/3999     14.2G   0.03635   0.07153  0.003802       403       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.84it/s]
                 all         14        150      0.244      0.187      0.169     0.0609

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  127/3999     14.2G   0.03814    0.0846  0.003568       576       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.50it/s]
                 all         14        150      0.257      0.182      0.164     0.0659

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  128/3999     14.2G    0.0399   0.07831  0.004013       452       640: 100% 3/3 [00:02<00:00,  1.18it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.96it/s]
                 all         14        150      0.186      0.246      0.145     0.0563

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  129/3999     14.2G   0.03868   0.08297  0.003955       486       640: 100% 3/3 [00:02<00:00,  1.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.99it/s]
                 all         14        150      0.187       0.28      0.153     0.0617

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  130/3999     14.2G   0.03691   0.06972  0.003265       421       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.48it/s]
                 all         14        150      0.271      0.244      0.185     0.0788

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  131/3999     14.2G   0.03762   0.08424  0.003576       496       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.54it/s]
                 all         14        150      0.235      0.241      0.173     0.0685

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  132/3999     14.2G   0.03635   0.07683  0.003609       522       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.09it/s]
                 all         14        150      0.285      0.237      0.158     0.0664

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  133/3999     14.2G   0.03733   0.07331  0.003646       441       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.61it/s]
                 all         14        150      0.212      0.216       0.13     0.0499

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  134/3999     14.2G    0.0385   0.08422   0.00363       522       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.53it/s]
                 all         14        150      0.213       0.22      0.163      0.062

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  135/3999     14.2G   0.03634   0.07753  0.003176       537       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.38it/s]
                 all         14        150       0.18      0.267      0.133     0.0469

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  136/3999     14.2G   0.03701    0.0788  0.002997       523       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.78it/s]
                 all         14        150        0.2      0.218      0.158     0.0668

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  137/3999     14.2G   0.03794   0.07358  0.002994       455       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.31it/s]
                 all         14        150      0.246      0.163      0.148     0.0579

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  138/3999     14.2G    0.0381   0.07784  0.003501       513       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.52it/s]
                 all         14        150      0.215       0.13      0.129     0.0504

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  139/3999     14.2G   0.03754   0.08408  0.003141       478       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.67it/s]
                 all         14        150      0.224      0.189       0.17     0.0613

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  140/3999     14.2G   0.03843   0.08562  0.003033       512       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.16it/s]
                 all         14        150      0.207      0.225      0.167     0.0671

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  141/3999     14.2G   0.03834   0.07882   0.00335       506       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.16it/s]
                 all         14        150      0.224      0.226      0.175     0.0688

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  142/3999     14.2G   0.03705   0.07211  0.003428       448       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.27it/s]
                 all         14        150      0.259      0.172      0.166     0.0655

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  143/3999     14.2G   0.03643   0.07713  0.003377       476       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.38it/s]
                 all         14        150      0.237      0.167      0.147     0.0584

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  144/3999     14.2G   0.03692   0.07941  0.003754       474       640: 100% 3/3 [00:02<00:00,  1.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.14it/s]
                 all         14        150      0.581      0.173      0.134     0.0506

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  145/3999     14.2G   0.03658   0.08377  0.003233       683       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.78it/s]
                 all         14        150      0.229      0.206      0.151     0.0527

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  146/3999     14.2G   0.03473   0.07883   0.00317       436       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.23it/s]
                 all         14        150      0.163      0.256      0.133     0.0501

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  147/3999     14.2G   0.03745   0.07596  0.003563       540       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.09it/s]
                 all         14        150      0.225      0.188      0.152     0.0598

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  148/3999     14.2G   0.03576   0.07779  0.003133       505       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.93it/s]
                 all         14        150      0.165       0.23      0.128     0.0503

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  149/3999     14.2G   0.03577   0.07039  0.003605       368       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.00it/s]
                 all         14        150      0.184      0.213      0.155     0.0618

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  150/3999     14.2G   0.03811    0.0843  0.003153       532       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.94it/s]
                 all         14        150      0.155      0.221      0.143     0.0631

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  151/3999     14.2G   0.03703   0.07383  0.003223       455       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.38it/s]
                 all         14        150      0.247      0.184      0.152     0.0656

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  152/3999     14.2G   0.03691     0.078  0.003195       474       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.78it/s]
                 all         14        150      0.296      0.158      0.158     0.0703

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  153/3999     14.2G    0.0359   0.07192  0.002904       518       640: 100% 3/3 [00:02<00:00,  1.17it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.81it/s]
                 all         14        150      0.216      0.212       0.16     0.0694

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  154/3999     14.2G   0.03589   0.07632  0.002872       535       640: 100% 3/3 [00:02<00:00,  1.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.19it/s]
                 all         14        150       0.23      0.176      0.152     0.0666

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  155/3999     14.2G   0.03646   0.07829  0.003299       556       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.06it/s]
                 all         14        150       0.19      0.208      0.141     0.0617

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  156/3999     14.2G   0.03531   0.07667  0.003285       495       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.40it/s]
                 all         14        150      0.556      0.178      0.139     0.0566

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  157/3999     14.2G   0.03537   0.07947   0.00308       499       640: 100% 3/3 [00:02<00:00,  1.11it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.71it/s]
                 all         14        150      0.186      0.197      0.126     0.0488

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  158/3999     14.2G   0.03433    0.0776  0.002946       535       640: 100% 3/3 [00:02<00:00,  1.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.26it/s]
                 all         14        150      0.184       0.23      0.166     0.0636

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  159/3999     14.2G   0.03402   0.07368  0.003283       381       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.34it/s]
                 all         14        150      0.235      0.207      0.183     0.0726

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  160/3999     14.2G   0.03233   0.07795  0.003214       487       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.18it/s]
                 all         14        150       0.23      0.248      0.185     0.0691

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  161/3999     14.2G   0.03667   0.07591  0.003197       471       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.28it/s]
                 all         14        150      0.238      0.231       0.16     0.0639

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  162/3999     14.2G   0.03416   0.07477  0.003155       486       640: 100% 3/3 [00:02<00:00,  1.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.63it/s]
                 all         14        150      0.263      0.234      0.182      0.074

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  163/3999     14.2G   0.03728   0.07809  0.003396       494       640: 100% 3/3 [00:02<00:00,  1.07it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.15it/s]
                 all         14        150      0.275      0.211      0.172     0.0678

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  164/3999     14.2G   0.03565   0.07787  0.003181       576       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.35it/s]
                 all         14        150      0.231       0.22      0.189      0.075

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  165/3999     14.2G   0.03369    0.0697  0.003188       496       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.13it/s]
                 all         14        150      0.199      0.187       0.14     0.0558

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  166/3999     14.2G   0.03496   0.07844  0.002953       443       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.89it/s]
                 all         14        150      0.216        0.2      0.141      0.055

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  167/3999     14.2G   0.03589   0.07786  0.003101       519       640: 100% 3/3 [00:02<00:00,  1.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.29it/s]
                 all         14        150       0.19      0.233      0.134     0.0531

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  168/3999     14.2G    0.0334   0.07536  0.002806       447       640: 100% 3/3 [00:02<00:00,  1.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.54it/s]
                 all         14        150      0.209      0.169       0.14     0.0538

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  169/3999     14.2G   0.03568    0.0805  0.002954       617       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.13it/s]
                 all         14        150       0.17      0.163      0.112     0.0461

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  170/3999     14.2G   0.03376   0.07106  0.003077       457       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.89it/s]
                 all         14        150      0.214      0.203      0.142     0.0603

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  171/3999     14.2G    0.0332   0.07611  0.003011       502       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.52it/s]
                 all         14        150      0.212      0.216      0.155     0.0708

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  172/3999     14.2G   0.03578   0.07509  0.003121       442       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.76it/s]
                 all         14        150      0.244      0.218      0.161     0.0608

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  173/3999     14.2G   0.03251   0.06532  0.002994       406       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.99it/s]
                 all         14        150      0.218      0.201      0.145     0.0553

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  174/3999     14.2G   0.03466   0.08043  0.002711       499       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.52it/s]
                 all         14        150      0.591      0.142      0.149     0.0549

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  175/3999     14.2G   0.03768    0.0705  0.003149       407       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.57it/s]
                 all         14        150      0.238      0.171      0.145     0.0519

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  176/3999     14.2G   0.03548   0.07507  0.003041       502       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.08it/s]
                 all         14        150      0.162      0.239      0.154     0.0602

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  177/3999     14.2G   0.03449   0.07346    0.0026       454       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.33it/s]
                 all         14        150      0.183      0.229      0.154     0.0625

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  178/3999     14.2G   0.03266   0.07249  0.003036       404       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.02it/s]
                 all         14        150      0.192      0.197      0.154     0.0556

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  179/3999     14.2G    0.0329   0.06794  0.003086       440       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.29it/s]
                 all         14        150      0.224      0.193      0.177     0.0661

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  180/3999     14.2G   0.03603   0.07284  0.002743       523       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.96it/s]
                 all         14        150      0.234       0.23      0.199     0.0803

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  181/3999     14.2G   0.03811   0.07432  0.002701       469       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.16it/s]
                 all         14        150      0.165       0.24      0.161     0.0649

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  182/3999     14.2G   0.03474   0.08858  0.002995       667       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.97it/s]
                 all         14        150      0.544      0.184       0.17     0.0659

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  183/3999     14.2G   0.03447   0.07515  0.002804       576       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.20it/s]
                 all         14        150      0.193      0.287      0.147     0.0587

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  184/3999     14.2G   0.03336   0.07254  0.002868       490       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.57it/s]
                 all         14        150      0.211      0.224      0.149     0.0612

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  185/3999     14.2G   0.03382   0.06551  0.002565       444       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.08it/s]
                 all         14        150      0.205      0.224      0.145     0.0552

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  186/3999     14.2G   0.03561   0.07753  0.003368       460       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.09it/s]
                 all         14        150      0.203      0.197      0.144     0.0574

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  187/3999     14.2G   0.03256   0.06888  0.003133       475       640: 100% 3/3 [00:03<00:00,  1.03s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.12it/s]
                 all         14        150      0.239      0.214       0.17     0.0642

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  188/3999     14.2G   0.03151   0.06746  0.002853       413       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.35it/s]
                 all         14        150      0.247      0.232      0.189     0.0729

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  189/3999     14.2G   0.03286   0.07051  0.003144       431       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.16it/s]
                 all         14        150      0.227      0.246       0.19     0.0743

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  190/3999     14.2G   0.03228   0.07408  0.002631       597       640: 100% 3/3 [00:02<00:00,  1.24it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.52it/s]
                 all         14        150      0.253      0.243      0.208     0.0818

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  191/3999     14.2G   0.03406   0.07237  0.002622       500       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.02it/s]
                 all         14        150      0.213      0.303      0.191      0.074

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  192/3999     14.2G   0.03265   0.07344  0.002737       449       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.99it/s]
                 all         14        150      0.209      0.244      0.168     0.0669

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  193/3999     14.2G   0.03237   0.06691  0.003005       448       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.77it/s]
                 all         14        150      0.194      0.237      0.149      0.059

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  194/3999     14.2G   0.03343   0.07932  0.002555       608       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.58it/s]
                 all         14        150      0.162       0.21      0.139     0.0544

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  195/3999     14.2G   0.03243    0.0772  0.002597       470       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.94it/s]
                 all         14        150      0.198       0.23      0.158      0.059

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  196/3999     14.2G   0.03185   0.06244  0.003065       401       640: 100% 3/3 [00:02<00:00,  1.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.34it/s]
                 all         14        150      0.217       0.19      0.155     0.0574

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  197/3999     14.2G   0.03246   0.06695  0.003044       389       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.01it/s]
                 all         14        150      0.223      0.248      0.167     0.0672

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  198/3999     14.2G   0.03571   0.06705  0.002949       473       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.43it/s]
                 all         14        150      0.211      0.204      0.157     0.0578

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  199/3999     14.2G   0.03393   0.07553   0.00311       463       640: 100% 3/3 [00:02<00:00,  1.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.15it/s]
                 all         14        150      0.262      0.181      0.161     0.0634

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  200/3999     14.2G   0.03271   0.06827  0.002592       418       640: 100% 3/3 [00:02<00:00,  1.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.33it/s]
                 all         14        150      0.181       0.21      0.152     0.0562

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  201/3999     14.2G   0.03615   0.07718  0.002433       404       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.41it/s]
                 all         14        150      0.151      0.203      0.137     0.0458

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  202/3999     14.2G   0.03356   0.07409   0.00259       360       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.22it/s]
                 all         14        150      0.188      0.152      0.112     0.0359

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  203/3999     14.2G   0.03139   0.06788  0.002841       361       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.24it/s]
                 all         14        150      0.148      0.177      0.116     0.0353

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  204/3999     14.2G   0.03371   0.06615   0.00253       417       640: 100% 3/3 [00:02<00:00,  1.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.64it/s]
                 all         14        150       0.15      0.199      0.128     0.0426

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  205/3999     14.2G    0.0313   0.06664  0.002534       387       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.51it/s]
                 all         14        150      0.162      0.182      0.114     0.0386

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  206/3999     14.2G   0.03289   0.06566  0.002408       457       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.51it/s]
                 all         14        150      0.185      0.198      0.137     0.0492

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  207/3999     14.2G   0.03267   0.06515  0.002756       431       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.39it/s]
                 all         14        150      0.192      0.193      0.137     0.0494

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  208/3999     14.2G   0.03422   0.07285   0.00294       546       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.05it/s]
                 all         14        150      0.194      0.186      0.139     0.0549

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  209/3999     14.2G   0.03252   0.06626  0.002826       401       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.08it/s]
                 all         14        150      0.179      0.253      0.154     0.0628

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  210/3999     14.2G   0.03264    0.0778  0.002862       550       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.83it/s]
                 all         14        150      0.177      0.255      0.179     0.0724

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  211/3999     14.2G   0.03582   0.07807  0.002714       520       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.14it/s]
                 all         14        150      0.279       0.16      0.178     0.0757

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  212/3999     14.2G   0.03128   0.07225  0.002809       485       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.31it/s]
                 all         14        150      0.608      0.158      0.169     0.0721

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  213/3999     14.2G   0.03116   0.06648  0.002671       405       640: 100% 3/3 [00:02<00:00,  1.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.01it/s]
                 all         14        150      0.178      0.209      0.159     0.0655

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  214/3999     14.2G    0.0311   0.07017  0.003016       430       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.14it/s]
                 all         14        150      0.198        0.2      0.144     0.0552

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  215/3999     14.2G   0.03502   0.07798  0.002876       652       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.64it/s]
                 all         14        150      0.166       0.18      0.128     0.0509

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  216/3999     14.2G   0.03069   0.06774  0.002814       430       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.51it/s]
                 all         14        150      0.144      0.232      0.141     0.0571

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  217/3999     14.2G   0.03256    0.0725  0.002727       553       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.77it/s]
                 all         14        150      0.198      0.204      0.148     0.0589

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  218/3999     14.2G   0.03105   0.06805  0.002888       406       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.89it/s]
                 all         14        150      0.168      0.218      0.132     0.0528

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  219/3999     14.2G   0.03369   0.06833  0.002433       408       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.29it/s]
                 all         14        150      0.516      0.243      0.147     0.0583

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  220/3999     14.2G   0.03278   0.06563  0.002558       331       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.95it/s]
                 all         14        150      0.515      0.254      0.156     0.0613

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  221/3999     14.2G    0.0332   0.07409  0.002531       575       640: 100% 3/3 [00:02<00:00,  1.06it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.16it/s]
                 all         14        150       0.17      0.242      0.142     0.0546

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  222/3999     14.2G    0.0328    0.0705  0.002275       566       640: 100% 3/3 [00:02<00:00,  1.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.50it/s]
                 all         14        150      0.153      0.226      0.145     0.0611

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  223/3999     14.2G   0.03224   0.07584  0.002389       530       640: 100% 3/3 [00:02<00:00,  1.17it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.90it/s]
                 all         14        150      0.196      0.184      0.144      0.059

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  224/3999     14.2G   0.03027   0.06387  0.002073       387       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.65it/s]
                 all         14        150      0.153       0.23      0.148     0.0562

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  225/3999     14.2G   0.03503   0.07697  0.002608       635       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.95it/s]
                 all         14        150      0.305      0.204      0.168     0.0634

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  226/3999     14.2G   0.03168   0.06866  0.002622       407       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.77it/s]
                 all         14        150      0.207      0.204      0.175     0.0715

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  227/3999     14.2G   0.03382   0.07738  0.002773       530       640: 100% 3/3 [00:02<00:00,  1.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.66it/s]
                 all         14        150      0.189      0.237      0.172     0.0748

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  228/3999     14.2G   0.03011   0.06911  0.002559       451       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.03it/s]
                 all         14        150      0.207      0.205      0.155     0.0609

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  229/3999     14.2G   0.03146   0.06943  0.002459       369       640: 100% 3/3 [00:02<00:00,  1.18it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.92it/s]
                 all         14        150      0.211      0.227      0.155     0.0629

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  230/3999     14.2G   0.03138   0.06209  0.002639       388       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.27it/s]
                 all         14        150      0.203       0.21      0.148     0.0674

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  231/3999     14.2G   0.03056   0.07217  0.002438       461       640: 100% 3/3 [00:02<00:00,  1.05it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.58it/s]
                 all         14        150      0.181      0.206      0.131     0.0527

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  232/3999     14.2G   0.03178   0.07369  0.002921       447       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.96it/s]
                 all         14        150      0.144      0.222      0.121     0.0468

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  233/3999     14.2G   0.03105   0.07161  0.002987       485       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.42it/s]
                 all         14        150      0.242      0.185      0.156     0.0598

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  234/3999     14.2G     0.031   0.06708  0.003061       369       640: 100% 3/3 [00:02<00:00,  1.28it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.96it/s]
                 all         14        150      0.221      0.183       0.14     0.0533

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  235/3999     14.2G   0.03028   0.06538   0.00243       379       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.75it/s]
                 all         14        150      0.232      0.177      0.142      0.051

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  236/3999     14.2G   0.03383   0.07849  0.002965       557       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.69it/s]
                 all         14        150      0.228      0.208      0.122     0.0419

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  237/3999     14.2G   0.03135   0.06768   0.00216       567       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.95it/s]
                 all         14        150      0.234      0.225      0.159     0.0548

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  238/3999     14.2G   0.03325   0.06947  0.002471       504       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.12it/s]
                 all         14        150      0.214      0.222      0.165     0.0632

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  239/3999     14.2G   0.03156   0.06755  0.002648       475       640: 100% 3/3 [00:02<00:00,  1.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.11it/s]
                 all         14        150       0.24      0.202      0.167     0.0684

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  240/3999     14.2G   0.03226   0.07513  0.002778       613       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.75it/s]
                 all         14        150      0.209       0.22       0.14     0.0573

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  241/3999     14.2G   0.03084   0.07444  0.003187       526       640: 100% 3/3 [00:02<00:00,  1.24it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.10it/s]
                 all         14        150      0.213       0.22      0.144     0.0574

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  242/3999     14.2G   0.02984   0.06826  0.002432       472       640: 100% 3/3 [00:02<00:00,  1.49it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.20it/s]
                 all         14        150      0.178      0.241      0.137     0.0536

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  243/3999     14.2G   0.02996   0.06492  0.002257       439       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.50it/s]
                 all         14        150      0.166       0.23      0.135     0.0486

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  244/3999     14.2G   0.03566   0.06949  0.002229       538       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.41it/s]
                 all         14        150      0.195      0.198      0.143     0.0549

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  245/3999     14.2G   0.03089   0.07021  0.002419       510       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.67it/s]
                 all         14        150      0.202      0.226      0.151     0.0564

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  246/3999     14.2G   0.03071   0.06211  0.002651       389       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.21it/s]
                 all         14        150      0.554      0.237      0.159     0.0613

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  247/3999     14.2G   0.03321   0.06761  0.002446       472       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.45it/s]
                 all         14        150      0.213      0.241      0.178     0.0748

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  248/3999     14.2G   0.03166   0.06957  0.002413       514       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.70it/s]
                 all         14        150      0.232      0.204      0.165     0.0688

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  249/3999     14.2G   0.02875   0.06775  0.002618       499       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.60it/s]
                 all         14        150      0.179      0.246      0.166     0.0676

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  250/3999     14.2G   0.03108   0.06709  0.002558       456       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.32it/s]
                 all         14        150      0.566      0.216      0.169     0.0717

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  251/3999     14.2G   0.03151    0.0739  0.002485       509       640: 100% 3/3 [00:02<00:00,  1.40it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.09it/s]
                 all         14        150      0.555      0.217      0.172     0.0763

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  252/3999     14.2G   0.03186   0.07398  0.002588       531       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.29it/s]
                 all         14        150      0.621      0.187      0.173     0.0743

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  253/3999     14.2G   0.02976   0.06722  0.002722       424       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.86it/s]
                 all         14        150      0.288      0.215      0.182     0.0797

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  254/3999     14.2G   0.03204   0.06845  0.002313       474       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.38it/s]
                 all         14        150      0.231      0.232      0.179     0.0735

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  255/3999     14.2G   0.03038   0.06902  0.002335       531       640: 100% 3/3 [00:02<00:00,  1.10it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.64it/s]
                 all         14        150      0.274      0.235      0.191     0.0825

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  256/3999     14.2G   0.03213   0.07496  0.002434       623       640: 100% 3/3 [00:02<00:00,  1.11it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.05it/s]
                 all         14        150      0.229      0.229      0.179     0.0777

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  257/3999     14.2G   0.03191   0.06197   0.00238       421       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.45it/s]
                 all         14        150      0.193      0.238      0.178     0.0804

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  258/3999     14.2G   0.03068    0.0724  0.002633       502       640: 100% 3/3 [00:02<00:00,  1.16it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.32it/s]
                 all         14        150      0.217      0.199       0.17     0.0679

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  259/3999     14.2G   0.02989   0.06942  0.002655       495       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.84it/s]
                 all         14        150       0.21      0.257      0.189     0.0886

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  260/3999     14.2G   0.02943    0.0634  0.002486       435       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.98it/s]
                 all         14        150       0.23      0.225      0.189      0.082

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  261/3999     14.2G   0.03245   0.07098  0.002192       520       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.15it/s]
                 all         14        150      0.196      0.222      0.173     0.0811

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  262/3999     14.2G   0.03108   0.06435  0.002188       440       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.00it/s]
                 all         14        150      0.207       0.19      0.157     0.0654

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  263/3999     14.2G   0.02963   0.06462  0.002824       371       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.00it/s]
                 all         14        150      0.237      0.182       0.16     0.0646

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  264/3999     14.2G    0.0311   0.06766  0.002327       495       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.27it/s]
                 all         14        150      0.248      0.183      0.152     0.0605

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  265/3999     14.2G    0.0323   0.06554  0.002227       347       640: 100% 3/3 [00:03<00:00,  1.01s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.55it/s]
                 all         14        150      0.227      0.206      0.147     0.0625

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  266/3999     14.2G   0.03151   0.06708  0.003037       447       640: 100% 3/3 [00:02<00:00,  1.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.15it/s]
                 all         14        150      0.187       0.22      0.145      0.058

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  267/3999     14.2G   0.03277   0.07251  0.002167       543       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.79it/s]
                 all         14        150      0.222      0.235      0.162     0.0706

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  268/3999     14.2G   0.02809   0.06559   0.00261       481       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.95it/s]
                 all         14        150      0.213      0.215      0.164     0.0709

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  269/3999     14.2G   0.03209   0.06901  0.002399       493       640: 100% 3/3 [00:02<00:00,  1.22it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.91it/s]
                 all         14        150      0.215      0.234      0.159     0.0634

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  270/3999     14.2G   0.02955   0.06778  0.002249       499       640: 100% 3/3 [00:02<00:00,  1.37it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.62it/s]
                 all         14        150      0.157       0.23      0.131     0.0559

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  271/3999     14.2G   0.02889    0.0663  0.002225       396       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.34it/s]
                 all         14        150      0.215      0.196      0.145     0.0585

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  272/3999     14.2G   0.03047   0.06824  0.002635       513       640: 100% 3/3 [00:01<00:00,  1.51it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.44it/s]
                 all         14        150       0.18      0.306      0.166     0.0671

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  273/3999     14.2G   0.02847   0.06178  0.002346       451       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.59it/s]
                 all         14        150      0.208      0.187      0.143      0.063

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  274/3999     14.2G   0.02983   0.07569  0.002735       585       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.67it/s]
                 all         14        150      0.239       0.19      0.157     0.0679

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  275/3999     14.2G   0.02842   0.06178  0.002506       377       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.94it/s]
                 all         14        150      0.239      0.232      0.173     0.0735

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  276/3999     14.2G   0.03096   0.06598  0.002438       521       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.29it/s]
                 all         14        150      0.196      0.235      0.181     0.0765

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  277/3999     14.2G   0.03166   0.06487  0.002525       528       640: 100% 3/3 [00:02<00:00,  1.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.46it/s]
                 all         14        150       0.16      0.279      0.164     0.0725

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  278/3999     14.2G   0.02937   0.06399  0.002401       386       640: 100% 3/3 [00:02<00:00,  1.15it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.66it/s]
                 all         14        150      0.204      0.254      0.144     0.0593

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  279/3999     14.2G    0.0304   0.07005  0.002289       537       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.39it/s]
                 all         14        150      0.269      0.178      0.158     0.0664

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  280/3999     14.2G   0.02981   0.06491  0.002434       512       640: 100% 3/3 [00:02<00:00,  1.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.90it/s]
                 all         14        150       0.26      0.198      0.155     0.0686

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  281/3999     14.2G   0.03123   0.06694  0.002469       332       640: 100% 3/3 [00:01<00:00,  1.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.69it/s]
                 all         14        150      0.222      0.212      0.162     0.0689

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  282/3999     14.2G   0.02753   0.06725  0.002008       507       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.07it/s]
                 all         14        150      0.181      0.242      0.148     0.0631

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  283/3999     14.2G   0.03039   0.06595  0.002752       366       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.25it/s]
                 all         14        150      0.189      0.236      0.161      0.064

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  284/3999     14.2G   0.03069   0.06211  0.002449       339       640: 100% 3/3 [00:02<00:00,  1.27it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.27it/s]
                 all         14        150      0.201      0.227      0.148      0.064

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  285/3999     14.2G   0.03091   0.06725  0.002642       505       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.53it/s]
                 all         14        150      0.193      0.197      0.141     0.0568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  286/3999     14.2G   0.02718   0.06115  0.002353       349       640: 100% 3/3 [00:01<00:00,  1.60it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.47it/s]
                 all         14        150      0.508       0.18      0.137     0.0582

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  287/3999     14.2G   0.02846   0.06968  0.002327       573       640: 100% 3/3 [00:02<00:00,  1.31it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.64it/s]
                 all         14        150      0.164      0.191      0.128     0.0529

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  288/3999     14.2G   0.02973     0.068  0.002584       553       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.15it/s]
                 all         14        150      0.164       0.16      0.126     0.0521

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  289/3999     14.2G   0.03254    0.0663   0.00238       439       640: 100% 3/3 [00:02<00:00,  1.08it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.00it/s]
                 all         14        150      0.152      0.269      0.135     0.0551

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  290/3999     14.2G   0.02915    0.0649  0.002251       538       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.50it/s]
                 all         14        150      0.169      0.234      0.128     0.0523

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  291/3999     14.2G   0.03037   0.06736  0.002363       472       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.24it/s]
                 all         14        150      0.151      0.267      0.142      0.065

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  292/3999     14.2G   0.03062   0.07348   0.00214       575       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.11it/s]
                 all         14        150      0.189      0.216      0.149     0.0641

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  293/3999     14.2G   0.03172   0.06896  0.002275       494       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.99it/s]
                 all         14        150      0.178      0.221      0.155     0.0629

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  294/3999     14.2G   0.02959   0.06641  0.002514       508       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.90it/s]
                 all         14        150      0.216      0.181       0.16     0.0633

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  295/3999     14.2G   0.03119   0.06105  0.002445       362       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.32it/s]
                 all         14        150      0.187        0.2      0.161     0.0665

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  296/3999     14.2G   0.02971   0.06801  0.002346       571       640: 100% 3/3 [00:02<00:00,  1.26it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.76it/s]
                 all         14        150       0.57      0.164      0.151     0.0582

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  297/3999     14.2G   0.03091   0.06922  0.002561       532       640: 100% 3/3 [00:01<00:00,  1.61it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.30it/s]
                 all         14        150      0.194      0.156      0.136     0.0537

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  298/3999     14.2G   0.02859    0.0648    0.0027       457       640: 100% 3/3 [00:02<00:00,  1.11it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.37it/s]
                 all         14        150      0.601      0.107      0.122     0.0506

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  299/3999     14.2G   0.02943   0.05912  0.001992       374       640: 100% 3/3 [00:02<00:00,  1.10it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.41it/s]
                 all         14        150      0.499      0.132       0.11     0.0434

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  300/3999     14.2G   0.03051   0.06684  0.002318       482       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.78it/s]
                 all         14        150      0.175      0.127      0.107     0.0432

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  301/3999     14.2G   0.03101   0.06934  0.002195       514       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.04it/s]
                 all         14        150      0.179      0.149      0.123      0.054

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  302/3999     14.2G   0.02933   0.07047  0.002302       694       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.50it/s]
                 all         14        150      0.199      0.168      0.127     0.0564

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  303/3999     14.2G    0.0301   0.06698  0.002084       529       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.24it/s]
                 all         14        150      0.208      0.182       0.14     0.0616

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  304/3999     14.2G   0.03033   0.06387   0.00223       353       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.50it/s]
                 all         14        150      0.213      0.182      0.135     0.0524

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  305/3999     14.2G   0.03112   0.06437   0.00235       400       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.28it/s]
                 all         14        150      0.184      0.177       0.13     0.0492

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  306/3999     14.2G   0.03117   0.07368  0.002213       638       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.09it/s]
                 all         14        150      0.211      0.186      0.144     0.0596

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  307/3999     14.2G   0.03051   0.06651  0.002304       549       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.16it/s]
                 all         14        150      0.182      0.207      0.159     0.0645

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  308/3999     14.2G   0.03138    0.0665  0.002465       616       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.27it/s]
                 all         14        150      0.179       0.23      0.149     0.0641

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  309/3999     14.2G   0.03263    0.0648  0.002839       440       640: 100% 3/3 [00:02<00:00,  1.48it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.82it/s]
                 all         14        150      0.178      0.202      0.154     0.0628

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  310/3999     14.2G    0.0284   0.06586  0.002275       460       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.42it/s]
                 all         14        150        0.2      0.205      0.168     0.0709

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  311/3999     14.2G   0.02876   0.06649  0.002203       440       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.03it/s]
                 all         14        150      0.219       0.16      0.154     0.0653

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  312/3999     14.2G   0.02765    0.0682  0.002128       496       640: 100% 3/3 [00:02<00:00,  1.35it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.86it/s]
                 all         14        150      0.196      0.161      0.134     0.0575

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  313/3999     14.2G    0.0284   0.06832  0.002096       552       640: 100% 3/3 [00:02<00:00,  1.19it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  5.41it/s]
                 all         14        150      0.141      0.239      0.124     0.0481

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  314/3999     14.2G   0.02762   0.07095  0.002487       527       640: 100% 3/3 [00:02<00:00,  1.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.22it/s]
                 all         14        150      0.174      0.213      0.134      0.056

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  315/3999     14.2G   0.02855    0.0609  0.002458       477       640: 100% 3/3 [00:02<00:00,  1.25it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.65it/s]
                 all         14        150      0.228      0.187      0.155     0.0683

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  316/3999     14.2G   0.02947   0.06064  0.002497       439       640: 100% 3/3 [00:02<00:00,  1.50it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.80it/s]
                 all         14        150      0.253      0.153      0.154     0.0666

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  317/3999     14.2G   0.02912   0.06867  0.002236       521       640: 100% 3/3 [00:02<00:00,  1.45it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.21it/s]
                 all         14        150      0.193      0.166      0.146     0.0635

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  318/3999     14.2G   0.03094   0.07085  0.002113       523       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.05it/s]
                 all         14        150      0.235      0.191      0.174     0.0748

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  319/3999     14.2G   0.03015   0.06773  0.002276       530       640: 100% 3/3 [00:02<00:00,  1.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.83it/s]
                 all         14        150      0.238      0.193      0.161     0.0716

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  320/3999     14.2G   0.02989   0.06571  0.002404       593       640: 100% 3/3 [00:02<00:00,  1.43it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.28it/s]
                 all         14        150      0.215      0.217      0.165     0.0726

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  321/3999     14.2G   0.03179   0.07255  0.002188       582       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.66it/s]
                 all         14        150      0.218      0.202      0.156     0.0676

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  322/3999     14.2G   0.02947   0.06519  0.002272       482       640: 100% 3/3 [00:02<00:00,  1.32it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.24it/s]
                 all         14        150      0.255      0.185      0.174     0.0645

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  323/3999     14.2G   0.02715   0.06451  0.002406       466       640: 100% 3/3 [00:02<00:00,  1.17it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.24it/s]
                 all         14        150      0.242      0.163      0.163     0.0721

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  324/3999     14.2G   0.02928   0.06748   0.00256       448       640: 100% 3/3 [00:03<00:00,  1.03s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.60it/s]
                 all         14        150      0.542      0.169      0.162     0.0648

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  325/3999     14.2G   0.02612   0.06095  0.002223       404       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.44it/s]
                 all         14        150       0.54      0.192      0.141     0.0542

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  326/3999     14.2G   0.03306     0.067  0.002028       529       640: 100% 3/3 [00:01<00:00,  1.54it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.56it/s]
                 all         14        150      0.174      0.205      0.148     0.0562

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  327/3999     14.2G   0.03056   0.06541   0.00241       508       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.49it/s]
                 all         14        150      0.577      0.157      0.155       0.06

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  328/3999     14.2G   0.02808   0.06285  0.001918       404       640: 100% 3/3 [00:01<00:00,  1.52it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.99it/s]
                 all         14        150      0.193      0.192       0.15      0.059

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  329/3999     14.2G   0.02972   0.06534  0.002311       527       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.20it/s]
                 all         14        150      0.172      0.188      0.138     0.0568

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  330/3999     14.2G   0.02876    0.0664   0.00213       516       640: 100% 3/3 [00:02<00:00,  1.30it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  5.02it/s]
                 all         14        150      0.256      0.164      0.147     0.0633

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  331/3999     14.2G   0.02872   0.06444  0.002482       461       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.54it/s]
                 all         14        150      0.252      0.178      0.168     0.0761

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  332/3999     14.2G   0.02905   0.06577  0.002103       523       640: 100% 3/3 [00:02<00:00,  1.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.51it/s]
                 all         14        150      0.208      0.187       0.15     0.0674

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  333/3999     14.2G    0.0291   0.06381  0.002342       477       640: 100% 3/3 [00:02<00:00,  1.15it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  5.25it/s]
                 all         14        150      0.211      0.167      0.129     0.0583

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  334/3999     14.2G   0.02746    0.0579  0.002248       437       640: 100% 3/3 [00:02<00:00,  1.21it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.75it/s]
                 all         14        150      0.205      0.145      0.145     0.0681

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  335/3999     14.2G   0.02931    0.0665  0.002167       449       640: 100% 3/3 [00:02<00:00,  1.42it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.84it/s]
                 all         14        150      0.163      0.158      0.127     0.0595

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  336/3999     14.2G   0.02642   0.06318  0.002018       414       640: 100% 3/3 [00:02<00:00,  1.38it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.75it/s]
                 all         14        150      0.171      0.172      0.118     0.0507

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  337/3999     14.2G   0.03052   0.06404  0.002213       493       640: 100% 3/3 [00:01<00:00,  1.57it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.16it/s]
                 all         14        150      0.204      0.155      0.133     0.0592

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  338/3999     14.2G   0.02966   0.06701  0.002156       497       640: 100% 3/3 [00:01<00:00,  1.56it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.59it/s]
                 all         14        150      0.225      0.164      0.135     0.0595

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  339/3999     14.2G   0.03051   0.06686  0.002294       531       640: 100% 3/3 [00:01<00:00,  1.59it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.35it/s]
                 all         14        150      0.171      0.192      0.137     0.0591

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  340/3999     14.2G   0.02711    0.0638  0.002133       446       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.48it/s]
                 all         14        150      0.227      0.171      0.151     0.0643

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  341/3999     14.2G   0.02844   0.07043  0.001833       604       640: 100% 3/3 [00:01<00:00,  1.58it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.16it/s]
                 all         14        150      0.257      0.184      0.167     0.0731

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  342/3999     14.2G   0.02733   0.06148  0.002062       460       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.13it/s]
                 all         14        150      0.158      0.208      0.161     0.0642

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  343/3999     14.2G    0.0268   0.06085  0.002161       384       640: 100% 3/3 [00:01<00:00,  1.55it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.24it/s]
                 all         14        150      0.578      0.172      0.164      0.067

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  344/3999     14.2G   0.02711   0.05812  0.001928       401       640: 100% 3/3 [00:02<00:00,  1.23it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.24it/s]
                 all         14        150      0.566      0.221      0.171     0.0636

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  345/3999     14.2G   0.03259   0.07102   0.00241       594       640: 100% 3/3 [00:02<00:00,  1.44it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.26it/s]
                 all         14        150      0.204      0.217      0.143     0.0549

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  346/3999     14.2G   0.02971   0.06784  0.002042       598       640: 100% 3/3 [00:02<00:00,  1.39it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.86it/s]
                 all         14        150      0.163      0.232      0.137     0.0572

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  347/3999     14.2G   0.02722   0.05912  0.002059       383       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.82it/s]
                 all         14        150      0.139      0.246      0.137     0.0584

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  348/3999     14.2G   0.02764   0.06124  0.002037       514       640: 100% 3/3 [00:02<00:00,  1.41it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.57it/s]
                 all         14        150      0.582      0.178       0.16      0.067

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  349/3999     14.2G   0.02994   0.06139  0.002147       518       640: 100% 3/3 [00:02<00:00,  1.34it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.56it/s]
                 all         14        150      0.239      0.185      0.149     0.0596

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  350/3999     14.2G   0.02948   0.06618  0.001854       569       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.83it/s]
                 all         14        150      0.201      0.203      0.158     0.0649

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  351/3999     14.2G   0.02896   0.06159  0.002226       426       640: 100% 3/3 [00:02<00:00,  1.33it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.19it/s]
                 all         14        150      0.236      0.226       0.18     0.0717

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  352/3999     14.2G   0.02851    0.0578  0.001841       371       640: 100% 3/3 [00:02<00:00,  1.24it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.64it/s]
                 all         14        150      0.225      0.238      0.165     0.0661

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  353/3999     14.2G   0.02884   0.07083  0.002015       698       640: 100% 3/3 [00:01<00:00,  1.62it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.38it/s]
                 all         14        150       0.19      0.226      0.151     0.0663

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  354/3999     14.2G   0.02836   0.05499   0.00239       461       640: 100% 3/3 [00:02<00:00,  1.36it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  4.66it/s]
                 all         14        150      0.204      0.226      0.157     0.0684

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  355/3999     14.2G   0.02872   0.06012  0.002168       480       640: 100% 3/3 [00:02<00:00,  1.47it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.40it/s]
                 all         14        150      0.196      0.198      0.144     0.0612

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  356/3999     14.2G   0.02894   0.06509  0.001871       507       640: 100% 3/3 [00:02<00:00,  1.46it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.88it/s]
                 all         14        150      0.229      0.183      0.159     0.0752

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  357/3999     14.2G   0.03004   0.06113  0.002272       424       640: 100% 3/3 [00:02<00:00,  1.29it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.76it/s]
                 all         14        150      0.216        0.2       0.16     0.0751

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  358/3999     14.2G   0.02907   0.06742  0.002224       451       640: 100% 3/3 [00:03<00:00,  1.04s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.02it/s]
                 all         14        150      0.167      0.218      0.148     0.0657

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  359/3999     14.2G   0.02659   0.05814  0.001891       421       640: 100% 3/3 [00:01<00:00,  1.53it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.83it/s]
                 all         14        150       0.17      0.207      0.157     0.0649
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 259, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

360 epochs completed in 0.338 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.5MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 213 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  2.46it/s]
                 all         14        150      0.211      0.257      0.189     0.0887
               metal         14         10          0          0     0.0108    0.00657
               other         14         64      0.291      0.219      0.142     0.0656
             plastic         14         76      0.341      0.553      0.414      0.194
Results saved to runs/train/exp
wandb: Waiting for W&B process to finish... (success).
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▃▄▅▄▄▆▇▇█▅▇▇▇▇▇▆▆▆▆▆█▅█▆▆▇██▇▇▇▇▆▇▇▇▆▆▇
wandb: metrics/mAP_0.5:0.95 ▁▂▃▄▃▃▅▆▆▇▄▆▆▆▇▆▆▆▅▅▆▇▄▇▆▅▇▇█▇▇▆▇▆▆▇▇▆▆▆
wandb:    metrics/precision ▁▆▆▇▇▂▇▇▃█▃▅▃▄▄▄▄▃▃▄▃▄▃▄▇▃▄▄▄▄▄▃▃▃▄▄▄▃▃▃
wandb:       metrics/recall ▅▃▆▁▂▃▄▅▇▆▃▇▅▄▄▄▄▄▅▃▇█▃▃▆▄▄▆▅▅▅▆▄▃▃▄▃▄▆▄
wandb:       train/box_loss █▅▅▄▄▃▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▁▂▁▂▁▁▁▁▁▁▁▁▁▁
wandb:       train/cls_loss █▇▅▄▄▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/obj_loss ▇██▇▇▅▆▅▄▄▃▃▃▃▄▄▃▃▃▂▃▂▂▃▂▂▂▂▂▂▁▂▁▂▂▂▂▂▁▁
wandb:         val/box_loss █▄▂▃▅▅▃▂▂▂▃▂▂▃▂▂▁▂▁▂▂▂▃▁▁▂▂▁▁▂▁▂▂▂▂▂▂▂▂▂
wandb:         val/cls_loss ▆▄▃▃▇█▄▂▁▂▅▃▄▅▃▅▂▄▃▅▆▃▅▃▅▇▅▄▃▄▅▄▆▄▅▄▅▇▅▅
wandb:         val/obj_loss ▁█▇▂▂▂▃▃▃▃▄▄▃▃▄▄▄▄▄▃▃▃▃▄▃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
wandb:                x/lr0 █▆▄▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr1 ▁▃▅▇████████████████████████████▇▇▇▇▇▇▇▇
wandb:                x/lr2 ▁▃▅▇████████████████████████████▇▇▇▇▇▇▇▇
wandb: 
wandb: Run summary:
wandb:           best/epoch 259
wandb:         best/mAP_0.5 0.18922
wandb:    best/mAP_0.5:0.95 0.08857
wandb:       best/precision 0.21019
wandb:          best/recall 0.25713
wandb:      metrics/mAP_0.5 0.18899
wandb: metrics/mAP_0.5:0.95 0.08868
wandb:    metrics/precision 0.21084
wandb:       metrics/recall 0.25713
wandb:       train/box_loss 0.02659
wandb:       train/cls_loss 0.00189
wandb:       train/obj_loss 0.05814
wandb:         val/box_loss 0.05777
wandb:         val/cls_loss 0.03319
wandb:         val/obj_loss 0.12038
wandb:                x/lr0 0.00911
wandb:                x/lr1 0.00911
wandb:                x/lr2 0.00911
wandb: 
wandb: Synced neat-elevator-1: https://wandb.ai/yolo-garbage-detection/YOLOv5/runs/1wgcbwdz
wandb: Synced 5 W&B file(s), 13 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20220818_023407-1wgcbwdz/logs
  ```
</details>

### Evidências do treinamento

Wandb Report: [Garbage Detection with YOLOv5](https://wandb.ai/yolo-garbage-detection/YOLOv5/reports/Garbage-Detection-with-YOLOv5--VmlldzoyNDkxMjY0?accessToken=b6zdixtj1zbg3y2p7v0grcfu8g2i2rm3zmehyb6ytzfdqosp5xyrodn69qfz9t17)

## Roboflow

Dataset: [Garbage Detection Dataset v1](https://app.roboflow.com/garbage-detection-t5mop/garbage-detection-gwzrk/1)

## HuggingFace

Demo app: [Garbage Detection](https://huggingface.co/spaces/afsm/garbage-detection)