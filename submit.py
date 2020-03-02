import fastai
import pandas as pd
import os.path as osp
from fastai.callbacks import SaveModelCallback
from fastai.vision import *

data = ImageList.from_folder('/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/病例资料的副本/data') \
    .split_none() \
    .label_empty() \
    .add_test_folder('test') \
    .transform(get_transforms(do_flip=True, max_rotate=40, max_zoom=2, max_warp=0.4), size=(490 // 2, 740 // 2)) \
    .databunch(bs=2) \
    .normalize(imagenet_stats)
learner = Learner(
    data,
    models.densenet121(num_classes=1),
    opt_func=optim.SGD,
    # loss_func=BCEWithLogitsFlat(),
    metrics=[root_mean_squared_error, mean_absolute_error, r2_score],
    model_dir='densenet121_regression',
    path='./models/'
)
learner.model = nn.DataParallel(learner.model, device_ids=[0, 1])

learner.load('densnet_best')

for img in data.test_ds:
    print(learner.predict(img[0]))
