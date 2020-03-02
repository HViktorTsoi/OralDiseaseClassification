import fastai
import pandas as pd
import os.path as osp
from fastai.callbacks import SaveModelCallback
from fastai.vision import *

from efficient_net import EfficientNet


def judge_is_path(row):
    return os.path.exists(row['path'])


def get_model():
    # return models.densenet121(num_classes=1)
    return EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)


root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/病例资料的副本/'

# 数据
# 特定牙面数据集
df_target = pd.read_csv(osp.join(root, 'target.csv'))
df_target['path'] = df_target['path'].map(lambda path: osp.join(root, 'images', *path.split('/')[-2:-1], path.split('/')[-1][:-4] + '.png'))
df_target = df_target.where(df_target.apply(judge_is_path, axis=1)).dropna(axis=0)

# 全体数据集
df = pd.read_csv(osp.join(root, 'result_all.csv'), encoding='GBK')
df['path'] = df['path'].map(lambda path: osp.join(root, 'images', *path.split('/')[:-1], path.split('/')[-1][:-4] + '.png'))

df = df.where(df.apply(judge_is_path, axis=1)).dropna(axis=0)
df['龋齿'] = df['龋齿'].map(lambda cls: min(cls, 1))
# df['牙结石'] = df['牙结石'].map(lambda cls: min(int(cls), 1))

# 选择特定部位的牙面
# df = df.merge(df_target, on='path', how='inner')

data = (ImageList
        .from_df(df=df, path='/', cols='path')
        .split_by_rand_pct(0.2, seed=1)
        .label_from_df(cols='牙结石', label_cls=FloatList)
        # .label_from_df(cols='牙菌斑', label_cls=CategoryList)
        # .label_from_df(cols='龋齿', label_cls=CategoryList)
        .transform(get_transforms(do_flip=True, max_rotate=40, max_zoom=2, max_warp=0.4, max_lighting=0.4), size=(240, 400))
        .databunch(bs=28, num_workers=8)
        .normalize(imagenet_stats))
print(data._label_list)

# 训练
learner = Learner(
    data,
    get_model(),
    opt_func=optim.SGD,
    metrics=[mean_absolute_error, r2_score, ],
    model_dir='ef_regression',
    path='./models/'
)
learner.model = nn.DataParallel(learner.model, device_ids=[0, 1])

learner.load('ef_best')

# learner.lr_find(wd=1e-4, end_lr=10)
# learner.recorder.plot()
# plt.show()

lr = 1e-4
learner.fit_one_cycle(100, max_lr=lr, div_factor=10, pct_start=0.1368, moms=(0.95, 0.85), wd=1e-4,
                      callbacks=[SaveModelCallback(learner, every='improvement', monitor='valid_loss', name='ef_best')])

# y_pred = learner.get_preds()
# print(y_pred)
