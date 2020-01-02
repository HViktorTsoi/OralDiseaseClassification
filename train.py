import fastai
import pandas as pd
import os.path as osp
from fastai.callbacks import SaveModelCallback
from fastai.vision import *

root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/病例资料的副本/'

# 数据
df = pd.read_csv(osp.join(root, 'result.csv'))
df['path'] = df['path'].map(lambda path: osp.join(root, 'images', *path.split('/')[2:-1], path.split('/')[-1][:-4] + '.png'))
data = (ImageList
        .from_df(df=df, path='/', cols='path')
        .split_by_rand_pct(0.2, seed=1)
        .label_from_df(cols='牙结石', label_cls=CategoryList)
        .transform(get_transforms(do_flip=True), size=(490 // 2, 740 // 2))
        .databunch(bs=64, num_workers=11)
        .normalize(imagenet_stats))

# 训练
learner = Learner(
    data,
    models.resnet152(),
    opt_func=optim.SGD,
    loss_func=CrossEntropyFlat(),
    metrics=[accuracy],
    model_dir='res152',
    path='./models/'
)
learner.model = nn.DataParallel(learner.model, device_ids=[0, 1, 2])
print(learner.loss_func)
learner.lr_find(wd=1e-4, end_lr=10)
learner.recorder.plot()
plt.show()

lr = 1e-2
learner.fit_one_cycle(200, max_lr=lr, div_factor=10, pct_start=0.1368, moms=(0.95, 0.85), wd=1e-4,
                      callbacks=[SaveModelCallback(learner, every='improvement', monitor='valid_loss', name='ef_best')])
