"""
Wrapper function for PyVista's decimate routine to simplify graphs given to Torch Geometric
"""


import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from scipy.spatial.transform import Rotation

import pyvista as pv 
import numpy as np

@functional_transform('decimation_face_to_edge')
class Decimation_FaceToEdge(BaseTransform):
    r"""Decimate and manipulate a mesh using PyVista and converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`decimation_face_to_edge`).

    Also adds a master node in the center of mass of the object.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
        target_reduction (float, default : 0.7): Approximate fraction of surface to decimate.
        traslate (bool, optional) : If set to :obj:`True`, traslate the mesh so that the center of mass is in the origin
        rotate (bool, optional) : If set to :obj:`True`, rotate the mesh in a random (phi,theta) polar angle
        scale (bool, optional) : If set to :obj:`True`, scale the mesh so that it fits in the [-1,1] cube around the origin
        

    """
    def __init__(self, remove_faces: bool = True, target_reduction: float = 0.7, traslate: bool = False, scale: bool = False):

        assert target_reduction < 1.0
        
        self.remove_faces = remove_faces
        self.target_reduction = target_reduction
        self.traslate = traslate
        self.scale = scale
        
    def __face_to_edge(self, data: Data) -> Data:
        if hasattr(data, 'face'):
            face = data.face
            edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

            data.edge_index = edge_index
            if self.remove_faces:
                data.face = None

        return data

    def __decimate(self, data: Data) -> Data:

        if hasattr(data,'face'):
            # face is (3,n_samples)
            n_faces = data.face.shape[1]
            vertices = data.pos.numpy()
            # pyvista wants the faces like
            # [[3 id1 id2 id3],
            #  [3 id1 id2 id3],...]
            # because they can be more general than triangles
            # but ours are all triangles to add a whole line of 3s 
            faces = np.hstack(((np.ones(n_faces,dtype=int)*3).reshape(-1,1),(data.face.T.numpy())))

            # create mesh
            mesh = pv.PolyData(vertices,faces)
            # now we decimate
            target_reduction = self.target_reduction
            mesh_decimated = mesh.decimate(target_reduction)

            # this is because in some versions of pyvista .is_all_triangles is not callable
            try:
                assert mesh_decimated.is_all_triangles()
            except:
                assert mesh_decimated.is_all_triangles

            center_of_mass = mesh_decimated.outline().center_of_mass()

            if self.traslate: # traslate so that the center of mass is in the origin
                mesh_decimated.translate(-center_of_mass,inplace=True)
                center_of_mass = center_of_mass*0.

            if self.scale: # scale so that it fits into [-1,1]^3 region
                scaling_factor = 1/max(abs(np.array(mesh_decimated.bounds)))
                mesh_decimated.scale(scaling_factor,inplace=True)


            # replace
            data.face = torch.from_numpy(np.copy(mesh_decimated.faces.reshape(-1,4)[:,1:].T))
            data.pos = torch.from_numpy(np.copy(mesh_decimated.points))

            return data, center_of_mass


    def forward(self, data: Data) -> Data:
        
        data,center_of_mass = self.__decimate(data)
        data = self.__face_to_edge(data)
        
        return data


@functional_transform('rotate_scale_traslate')
class RotateScaleTraslate(BaseTransform):
    r"""Decimate and manipulate a mesh using PyVista and converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]` (functional name: :obj:`decimation_face_to_edge`).

    Args:
        traslate (bool, optional) : If set to :obj:`True`, traslate the mesh so that the center of mass is in the origin
        rotate (bool, optional) : If set to :obj:`True`, rotate the mesh in a random (phi,theta) polar angle
        scale (bool, optional) : If set to :obj:`True`, scale the mesh so that it fits in the [-1,1] cube around the origin
    """
    def __init__(self, traslate: bool = False, scale: bool = False, rotate: bool = False):

        
        self.traslate = traslate
        self.scale = scale
        self.rotate = rotate
        

    def __manipulate(self, data: Data) -> Data:

        # data should have .pos attribute that represents the vertices cartesian coordinates
        # also a edge_index that represents edges but not important

        if self.rotate:

            rotation_matrix = torch.from_numpy(Rotation.random().as_matrix()).float()

            data.pos = data.pos @ rotation_matrix


        return data

    def forward(self, data: Data) -> Data:
        
        return self.__manipulate(data)


