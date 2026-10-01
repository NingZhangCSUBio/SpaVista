import torch
import numpy as np
from torchvision.transforms import functional as TF
import os
import glob
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _resolve_dataset_root(data_root, directory_name):
    """Accept either a dataset directory or a parent containing it."""
    nested_root = os.path.join(data_root, directory_name)
    return nested_root if os.path.isdir(nested_root) else data_root


class ViT_HER2ST(torch.utils.data.Dataset):
    """HER2ST dataset for patient-level cross-validation."""
    def __init__(self,train=True,gene_list=None,fold=0,test_slice=0,patch_size=224,task_type='HEG',train_aug=True,normal=True,data_root='data'):
        super(ViT_HER2ST, self).__init__()

        self.choosegene=task_type
        self.train_aug=train_aug
        self.normal=normal
        dataset_root = _resolve_dataset_root(data_root, 'her2st')

        if self.choosegene=='HEG':
            self.cnt_dir = os.path.join(dataset_root, 'ST-cnts_250')
        elif self.choosegene=='HVG':
            self.cnt_dir = os.path.join(dataset_root, 'ST-cnts_785')
        else:
            print('The gene list is wrong!')

        self.img_dir = os.path.join(dataset_root, 'data/ST-imgs')
        self.pos_dir = os.path.join(dataset_root, 'data/ST-spotfiles')
        self.lbl_dir = os.path.join(dataset_root, 'data/ST-pat/lbl')
        self.r=patch_size//2

        if self.choosegene=='HEG':
            gene_list = list(np.load(os.path.join(dataset_root, 'genes_her2st_250.npy'),allow_pickle=True))
        else:
            gene_list = list(np.load(os.path.join(dataset_root, 'genes_her2st_hvg785.npy'),allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]  # The first two characters identify each sample.
        self.train = train

        if self.choosegene=='HVG':
            samples = names[1:33]  # Use samples 2 through 32 for the HVG setting.
        else:
            samples = names

        te_names=self.get_samples_by_fold(samples, fold)
        tr_names = list(set(samples)-set(te_names))
        tr_names.sort()

        if train:
            names = tr_names
            print('Training the slides:',names)
        else:
            names = te_names
            names = [names[test_slice]]
            print('Testing the slides:',names)

        self.exp_list=[]
        self.center_list=[]
        self.loc_list=[]
        self.img_list=[]
        print('Loading data...')
        self.gene_set = list(gene_list)
        for i in names:
            meta_info=self.get_meta(i)
            exp_info=scp.transform.log(scp.normalize.library_size_normalize(meta_info[self.gene_set].values))
            center_info=np.floor(meta_info[['pixel_x','pixel_y']].values).astype(int)
            loc_info=meta_info[['x','y']].values
            self.exp_list.append(exp_info)
            self.center_list.append(center_info)
            self.loc_list.append(loc_info)

            img_info=np.array(self.get_img(i))


            for x,y in center_info:
                patch = img_info[(y-self.r):(y+self.r),(x-self.r):(x+self.r),:]
                self.img_list.append(patch)
        self.exp_list=np.concatenate(self.exp_list, axis=0)
        self.center_list=np.concatenate(self.center_list, axis=0)
        self.loc_list=np.concatenate(self.loc_list, axis=0)
        print(self.exp_list.shape,self.center_list.shape,self.center_list.shape)


    def __getitem__(self, index):

        patch=self.img_list[index]  # Image patch.
        patch=self.transform(patch)
        center=self.center_list[index]  # Pixel coordinates.
        position= self.loc_list[index]  # Array coordinates.
        exp=self.exp_list[index]  # Gene expression.
        exp = torch.Tensor(exp)
        position=torch.LongTensor(position)

        if self.train:
            return patch, position, exp
        else:
            return patch, position, exp, torch.Tensor(center)

    def __len__(self):
        return len(self.exp_list)

    def get_samples_by_fold(self,samples, fold):
        'select samples by fold patient'

        fold_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        selected_letter = [letter for letter, f in fold_mapping.items() if f == fold]
        selected_samples = [sample for sample in samples if sample[0] in selected_letter]

        return selected_samples

    def transform(self, image):
        image = Image.fromarray(image)

        if self.train and self.train_aug:
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)


            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)

        # Convert to tensor
        image = TF.to_tensor(image)

        # Normalize using ImageNet mean and std
        if self.normal:
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):

        path = self.cnt_dir+'/'+name+'_sub.parquet'
        df = pd.read_parquet(path)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i]))
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')),how='inner')

        return meta


class ViT_SKIN(torch.utils.data.Dataset):
    """cSCC dataset for patient-level cross-validation."""
    def __init__(self,train=True,gene_list=None,fold=0,test_slice=0,patch_size=224,task_type='HEG',train_aug=True,normal=True,data_root='data'):
        super(ViT_SKIN, self).__init__()

        dataset_root = _resolve_dataset_root(data_root, 'skin')
        self.dir = os.path.join(dataset_root, 'GSE144240_RAW/')
        self.r=patch_size//2
        self.choosegene=task_type
        self.train_aug=train_aug
        self.normal=normal

        if self.choosegene=='HEG':
            self.cnt_dir = os.path.join(dataset_root, 'skin_ST-cnts_250')
        elif self.choosegene=='HVG':
            self.cnt_dir = os.path.join(dataset_root, 'skin_ST-cnts_171')
        else:
            raise ValueError("task_type must be 'HEG' or 'HVG'.")

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)

        if self.choosegene=='HEG':
            gene_list = list(np.load(os.path.join(dataset_root, 'genes_skin_250.npy'),allow_pickle=True))
        elif self.choosegene=='HVG':
            gene_list = list(np.load(os.path.join(dataset_root, 'skin_hvg_cut_1000.npy'),allow_pickle=True))
        else:
            print('The gene list is wrong!')
        self.gene_list = gene_list
        self.train = train

        samples = names
        te_names = samples[fold*3:(fold+1)*3]
        tr_names=[item for item in samples if item not in te_names]

        if train:
            names = tr_names
            print('Training the slides:',names)
        else:
            names = te_names
            names = [names[test_slice]]
            print('Testing the slides:',names)

        self.exp_list=[]
        self.center_list=[]
        self.loc_list=[]
        self.img_list=[]
        self.ll_list=[]
        print('Loading data...')
        self.gene_set = list(gene_list)
        for i in names:
            meta_info=self.get_meta(i)
            exp_info=scp.transform.log(scp.normalize.library_size_normalize(meta_info[self.gene_set].values))
            center_info=np.floor(meta_info[['pixel_x','pixel_y']].values).astype(int)
            loc_info=meta_info[['x','y']].values
            self.exp_list.append(exp_info)
            self.center_list.append(center_info)
            self.loc_list.append(loc_info)

            img_info=np.array(self.get_img(i))


            for x,y in center_info:
                patch = img_info[(y-self.r):(y+self.r),(x-self.r):(x+self.r),:]
                self.img_list.append(patch)
        self.exp_list=np.concatenate(self.exp_list, axis=0)
        self.center_list=np.concatenate(self.center_list, axis=0)
        self.loc_list=np.concatenate(self.loc_list, axis=0)
        print(self.exp_list.shape,self.center_list.shape,self.center_list.shape)

    def transform(self, image):
        image = Image.fromarray(image)

        if self.train and self.train_aug:
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)


            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)

        # Convert to tensor
        image = TF.to_tensor(image)

        # Normalize using ImageNet mean and std
        if self.normal:
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image


    def __getitem__(self, index):

        patch=self.img_list[index]  # Image patch.
        patch=self.transform(patch)
        center=self.center_list[index]  # Pixel coordinates.
        position= self.loc_list[index]  # Array coordinates.
        exp=self.exp_list[index]  # Gene expression.
        exp = torch.Tensor(exp)
        position=torch.LongTensor(position)

        if self.train:
            return patch, position, exp
        else:
            return patch, position, exp, torch.Tensor(center)

    def __len__(self):
        return len(self.exp_list)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = os.path.join(self.cnt_dir, name + '_sub.parquet')
        df = pd.read_parquet(path)
        return df


    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i]))
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta
