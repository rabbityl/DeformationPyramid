import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

from skimage import io
import os
from .geometry import  depth_2_pc



def opencv_to_pytorch3d(T):
    ''' ajust axis
    :param T: 4x4 mat
    :return:
    '''
    origin = np.array(((-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
    origin = torch.from_numpy(origin).float().to(T)
    return T @ origin


class PCDRender(nn.Module):

    def __init__(self, K, img_size=(500,512), device=torch.device("cuda:0")):
        super().__init__()

        self.camera = None
        self.device = device

        self.img_size = img_size # (width, height) of the image

        self.camera = self.init_camera(K)

        raster_settings = PointsRasterizationSettings( image_size=self.img_size, radius = 0.005, points_per_pixel = 10)

        self.rasterizer = PointsRasterizer(cameras=self.camera, raster_settings=raster_settings).to(device)

        self.compositor= AlphaCompositor().to(device)


    def load_pcd (self, pcd) :
        feature = torch.ones_like(pcd)
        point_cloud = Pointclouds(points=[pcd], features=[feature]).to(self.device)
        return point_cloud

    def init_camera(self, K, T=torch.eye(4),  ):

        # T = T.to( self.device)
        T_py3d =  opencv_to_pytorch3d(T).to(device)
        R = T_py3d[:3, :3]
        t = T_py3d[:3, 3:]

        """creat camera"""
        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
        img_size = self.img_size

        camera = PerspectiveCameras( R=R[None], T=t.T, focal_length=f, principal_point=p, in_ndc=False, image_size=(img_size,)).to(device)

        return camera



    def forward(self, point_clouds) -> torch.Tensor:

        point_clouds = self.load_pcd(point_clouds)

        fragments = self.rasterizer(point_clouds, gamma=(1e-5,))

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius


        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            gamma=(1e-5,),
        )

        images = images.permute(0, 2, 3, 1)

        return images, fragments.zbuf



if __name__ == '__main__':


    """data path"""
    seq_name = "moose6OK9_AttackTrotRM"
    seq_dir = "/home/liyang/workspace/NeuralTracking/GlobalReg/example/" + seq_name
    depth_name = "cam1_0015.png"
    intr_name = "cam1intr.txt"
    tgt_depth_image_path = os.path.join( seq_dir,"depth", "cam1_0009.png")
    intrinsics_path = os.path.join(seq_dir, intr_name)
    K = np.loadtxt(intrinsics_path)
    tgt_depth = io.imread( tgt_depth_image_path )/1000.
    tgt_pcd = depth_2_pc(tgt_depth,K).transpose(1,2,0)
    tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float()
    # tgt_pcd = tgt_pcd - tgt_pcd.mean(dim=0, keepdim=True)



    renderer = PCDRender(K)

    img = renderer.render_pcd(tgt_pcd)

    plt.imshow(img[0, ..., :3].cpu().numpy())
    plt.show()

