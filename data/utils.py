import igl
import yaml
from yaml import CLoader as Loader, CDumper as Dumper

import os
import os.path as osp
import glob
import yaml
import igl
import numpy as np
import torch
from torch_geometric.data import (Data, InMemoryDataset)
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
    
def read_model(obj_path, feat_path):
    m = {}
    m["vertices"], _, m["normals"], m["face_indices"], _, m["normal_indices"] = igl.read_obj(obj_path)
            
    with open(feat_path) as fi:
        m["features"] = yaml.load(fi, Loader=Loader)
    return m


def get_averaged_normals(model):
    n = model["normals"]
    ni = model["normal_indices"]
    v = model["vertices"]
    f = model["face_indices"]
    normals = {}

    for i in range(f.shape[0]):
        for j in range(3):
            vert = f[i][j]
            norm = n[ni[i, j]]

            if vert not in normals:
                normals[vert] = []
            
            normals[vert].append(norm)
    
    av_normals = np.zeros((v.shape[0], 3))

    cnt = 0
    for v in sorted(normals):
        av_normals[cnt] = np.mean(np.array(normals[v]), axis=0)
        cnt += 1
    return av_normals

class ABCDataset(InMemoryDataset):
    r""" The ABC dataset from the `"ABC: A Big CAD Model Dataset for Geometric Deep Learning"
    <https://deep-geometry.github.io/abc-dataset/>`_paper, containing about 1M CAD models
    with ground truth for surface normals, patch segmentation and sharp features.

    Args:
        root (string): Root directory where the dataset should be saved.
        categories (string or [string], optional): The categories of the CAD
            model ground truth values. Can be one of 'Normals', 'Patches', 'Curves'.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deep-geometry.github.io/abc-dataset/'

    def __init__(self, root, typ="Curves", train=True, transform=None, pre_transform=None, pre_filter=None):
        super(ABCDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[4]
        self.typ = typ
        if self.typ == "Edges":
            p = path
        if self.typ == "Normals":
            p = path.replace(".pt", "_normals.pt")
        if self.typ == "Types":
            p = path.replace(".pt", "_types.pt")

        self.data, self.slices = torch.load(p)
        from yaml import CLoader as Loader, CDumper as Dumper

        
        #torch.load(self.collate(train_data), self.processed_paths[0])
        #torch.save(self.collate(test_data), self.processed_paths[3])
        if train:
            with open(self.processed_paths[1], "r") as f:
                self.faces = yaml.load(f, Loader=Loader)
            with open(self.processed_paths[2], "r") as f:
                self.names = yaml.load(f, Loader=Loader)
            with open(self.processed_paths[3], "r") as f:
                self.fpatches = yaml.load(f, Loader=Loader)
        else:
            with open(self.processed_paths[5], "r") as f:
                self.faces = yaml.load(f, Loader=Loader)
            with open(self.processed_paths[6], "r") as f:
                self.names = yaml.load(f, Loader=Loader) 
            with open(self.processed_paths[7], "r") as f:
                self.fpatches = yaml.load(f, Loader=Loader) 

    @property
    def raw_file_names(self):
        return [
            'train_data', 'train_label', 'test_data', 'test_label'
        ]

    @property
    def processed_file_names(self):
        #cats = '_'.join([cat[:3].lower() for cat in self.categories])
        fns = []
        for n in ["train", "test"]:
            for c in ['data', "faces", "names", "f_patches"]:
                fns.append('{}_{}.pt'.format(n, c))
        return fns
    
    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        data = self.data
        return data.y.max().item() + 1 if data.y.dim() == 1 else data.y.size(1)

#    def download(self):
#        for name in self.raw_file_names:
#            url = '{}/{}.zip'.format(self.url, name)
#            path = download_url(url, self.raw_dir)
#            extract_zip(path, self.raw_dir)
#            os.unlink(path)

    def process_raw_path(self, data_path, label_path):
        data_list = []
        obj_paths = sorted(glob.glob(osp.join(data_path, '*.obj')))
        feat_paths = sorted(glob.glob(osp.join(label_path, '*.yml')))
        
        points = []
        normals = []
        patches = []
        edges = []
        faces = []
        names = []
        types = []
        face_patches = []
        t_map = {"Plane": 0, "Cylinder": 1, "Cone": 2, "Sphere": 3, "Torus": 4, "Bezier": 5, "BSpline": 6, "Revolution": 7,"Extrusion": 8, "Other": 9}
        cnt = 0
        for idx, obj in enumerate(obj_paths):
            if cnt == 10:
                break
            
            if os.path.getsize(obj) >= 10 * 1024**2 or os.path.getsize(feat_paths[idx+5]) >= 10 * 1024**2:
                #print("Skipping large file", obj)
                continue
            print(idx, cnt)
            m = read_model(obj, feat_paths[idx])
            normal = get_averaged_normals(m)
            #print(normal.shape)
            
            patch = np.zeros(m["vertices"].shape[0], dtype=np.long)
            edge = np.zeros(m["vertices"].shape[0], dtype=np.long)
            typ = np.zeros(m["vertices"].shape[0], dtype=np.long)
            f_patch = np.zeros(m["face_indices"].shape[0], dtype=np.long)
            invalid = False
            for i, fe in enumerate(m["features"]["surfaces"]):
                if invalid:
                    break
                t = t_map[fe["type"]]
                for j in fe["vert_indices"]:
                    if j >= patch.shape[0]:
                        invalid = True
                        break
                    patch[j] = i
                    typ[j] = t
                for j in fe["face_indices"]:
                    f_patch[j] = i
                                  
            if invalid:
                #print("Skipping model %s"%obj)
                continue

            for i, fe in enumerate(m["features"]["curves"]):
                val = -1
                if fe["sharp"]:
                    val = -2
                    
                for j in range(len(fe["vert_indices"])):
                    v_s = fe["vert_indices"][j]
                    typ[v_s] = val
                    if val == -2:
                        edge[v_s] = 1
            points.append(torch.tensor(m["vertices"].astype(np.float32)).squeeze())
            normals.append(torch.tensor(normal.astype(np.float32)).squeeze())
            patches.append(torch.tensor(patch).squeeze())
            types.append(torch.tensor(typ).squeeze())
            edges.append(torch.tensor(edge).squeeze())
            faces.append(m["face_indices"])
            face_patches.append(f_patch)
            names.append(obj)
            cnt += 1

        cnt = 0
        for (v, n, p, t, e) in zip(points, normals, patches, types, edges):
            cnts = torch.tensor(np.ones(p.shape, dtype=np.long)*cnt)
            cnt += 1
            #if self.typ == "Curves":
            #data = Data(pos=v, idx=cnts, y=e)
            data = Data(pos=v, idx=cnts, y=t)
            #data = Data(pos=v, idx=cnts, y=n)
#           data = Data(pos=v, idx=cnts, y=n)#y_typ=t, y_patch=p, y_normal=n, y_edge=e)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list, faces, names, face_patches

    def process(self):
        train_data, train_faces, train_names, train_fpatches = self.process_raw_path(*self.raw_paths[0:2])
        test_data, test_faces, test_names, test_fpatches = self.process_raw_path(*self.raw_paths[2:4])

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(test_data), self.processed_paths[4])
        with open(self.processed_paths[1], "w") as f:
            #tta = np.hstack([np.vstack(train_faces), np.hstack(train_fpatches).reshape(-1, 1)])
            yaml.dump(train_faces, f)
        with open(self.processed_paths[2], "w") as f:
            yaml.dump(train_names, f)
        with open(self.processed_paths[3], "w") as f:
            yaml.dump(train_fpatches, f)
        with open(self.processed_paths[5], "w") as f:
            #tta = np.hstack([np.vstack(test_faces), np.hstack(test_fpatches).reshape(-1, 1)])
            yaml.dump(test_faces, f)
        with open(self.processed_paths[6], "w") as f:
            yaml.dump(test_names, f)   
        with open(self.processed_paths[7], "w") as f:
            yaml.dump(test_fpatches, f)

    def __repr__(self):
        return '{}({}, categories={})'.format(self.__class__.__name__,
                                              len(self), self.categories)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])



