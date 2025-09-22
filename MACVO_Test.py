import torch
from Utility.Config import load_config
from pathlib import Path
from DataLoader import StereoData
import pypose as pp
import numpy as np
import Module
import cv2
from matplotlib import pyplot as plt
from Utility.Visualize import fig_plt, rr_plt
from Utility.Point import filterPointsInRange, pixel2point_NED
import rerun as rr
import os

def load_image(image_path) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (704, 704), interpolation=cv2.INTER_LINEAR)  # Resize to 704x704
    result = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    result /= 255.
    npy_path = os.path.splitext(os.path.basename(image_path))[0] + ".npy"
    np.save(npy_path, result.squeeze(0).cpu().numpy())
    return result

def load_stereo_frame(image_path_left: str, image_path_right: str) -> StereoData:
    T_BS = pp.identity_SE3(1)        # or pp.SE3(torch.randn(B, 7))
    
    fx = 1847.5905420747683 / 4
    fy = 1847.5905420747683 / 4
    cx = 1391.3 / 4
    cy = 1407.177 / 4

    K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]]).unsqueeze(0)
    
    baseline = torch.tensor([0.17007674086397787])                  # baseline in meters

    time_ns = [123456789]                             # list of timestamps
    height = 704
    width = 704
    imageL = load_image(image_path_left)
    imageR = load_image(image_path_right)

    return StereoData(
        T_BS=T_BS,
        K=K,
        baseline=baseline,
        time_ns=time_ns,
        height=height,
        width=width,
        imageL=imageL,
        imageR=imageR
    )

if __name__ == "__main__":
    cfg, cfg_dict = load_config(Path("Config/Experiment/MACVO/MACVO_Fast.yaml"))
    odomcfg = cfg.Odometry
    edge_width = 30
    match_cov_default = 0.25
    # Initialize modules for VO
    Frontend = Module.IFrontend.instantiate(odomcfg.frontend.type, odomcfg.frontend.args)
    KeypointSelector = Module.IKeypointSelector.instantiate(odomcfg.keypoint.type, odomcfg.keypoint.args)
    ObserveCovModel  = Module.ICovariance2to3.instantiate(odomcfg.cov.obs.type, odomcfg.cov.obs.args)
    image_path = "C:/Users/mch/Documents/test/MACVO/inputs/5/"
    frame0 = load_stereo_frame(image_path + "left_t1.jpg", image_path + "right_t1.jpg")
    frame1 = load_stereo_frame(image_path + "left_t2.jpg", image_path + "right_t2.jpg")
    depth0 = Frontend.estimate_depth(frame0)
    depth1, match01 = Frontend.estimate_pair(frame0, frame1)

    depth_0_cpu = depth0.depth.squeeze().cpu().numpy()  # Convert to numpy array for visualization
    cov_0_cpu = depth0.cov.squeeze().cpu().numpy()  # Convert to numpy array for visualization
    depth_1_cpu = depth1.depth.squeeze().cpu().numpy()  # Convert to numpy array for visualization
    cov_1_cpu = depth1.cov.squeeze().cpu().numpy()  # Convert to numpy array for visualization
    
    kp0_uv = KeypointSelector.select_point(frame0,2000, depth0, depth1, match01)
    kp1_uv  = kp0_uv + Frontend.retrieve_pixels(kp0_uv, match01.flow).T
    
    inbound_mask= filterPointsInRange(
            kp1_uv, 
            (edge_width, frame1.width - edge_width), 
            (edge_width, frame1.height - edge_width)
        )
    
    kp0_uv  = kp0_uv[inbound_mask]
    kp1_uv  = kp1_uv[inbound_mask]

    # Retrieve depth and depth cov for kp on frame 0 and 1 ##########################
    kp0_d               = Frontend.retrieve_pixels(kp0_uv, depth0.depth).squeeze(0)
    kp0_disparity       = Frontend.retrieve_pixels(kp0_uv, depth0.disparity)
    kp0_sigma_disparity = Frontend.retrieve_pixels(kp0_uv, depth0.disparity_uncertainty)
    kp0_sigma_dd        = Frontend.retrieve_pixels(kp0_uv, depth0.cov)
    kp0_sigma_dd        = kp0_sigma_dd.squeeze(0) if kp0_sigma_dd is not None else None

    kp1_d               = Frontend.retrieve_pixels(kp1_uv, depth1.depth).squeeze(0)
    kp1_disparity       = Frontend.retrieve_pixels(kp1_uv, depth1.disparity)
    kp1_sigma_disparity = Frontend.retrieve_pixels(kp1_uv, depth1.disparity_uncertainty)
    kp1_sigma_dd        = Frontend.retrieve_pixels(kp1_uv, depth1.cov)
    kp1_sigma_dd        = kp1_sigma_dd.squeeze(0) if kp1_sigma_dd is not None else None

    # Retrieve match cov for kp on frame 0 and 1    #################################
    num_kp = kp0_uv.size(0)
    
    # kp 0 has a fake sigma uv as it is manually selected pixels. This UV 
    # represents the uncertainty introduced by the quantization process when 
    # taking photo with discrete pixels.
    kp0_sigma_uv = torch.ones((num_kp, 3), device=torch.device("cuda")) * match_cov_default
    kp0_sigma_uv[..., 2] = 0.   # No sigma_uv off-diag term.
    
    kp1_sigma_uv = Frontend.retrieve_pixels(kp0_uv, match01.cov)
    kp1_sigma_uv = kp1_sigma_uv.T if kp1_sigma_uv is not None else None
    
    # Record color of keypoints (for visualization) #################################
    kp0_uv_cpu = kp0_uv.cpu()
    kp0_color  = frame0.imageL[..., kp0_uv_cpu[..., 1], kp0_uv_cpu[..., 0]].squeeze(0).T
    kp0_color  = (kp0_color * 255).to(torch.uint8)
    
    # Project from 2D -> 3D #########################################################
    pos0_Tc = pixel2point_NED(kp0_uv, kp0_d, frame0.frame_K).cpu()
    pos0_covTc  = ObserveCovModel.estimate(frame0, kp0_uv, depth0, kp0_sigma_dd, kp0_sigma_uv).cpu()
    pos1_covTc  = ObserveCovModel.estimate(frame1, kp1_uv, depth1, kp1_sigma_dd, kp1_sigma_uv).cpu()
    kp1_uv_cpu = kp1_uv.cpu()
    for i in range(len(pos0_covTc)):
        print("Point:", kp1_uv_cpu[i])
        print("Cov0:", pos0_covTc.squeeze(0)[i])
        print("Cov1:", pos1_covTc.squeeze(0)[i])
        print("-------")
    # np.save(image_path + "kp0", kp0_uv.cpu().squeeze().numpy())
    # np.save(image_path + "kp1", kp1_uv.cpu().squeeze().numpy())
    # np.save(image_path + "pos0_covTc", pos0_covTc.squeeze().numpy())
    # np.save(image_path + "pos1_covTc", pos1_covTc.squeeze().numpy())
    # print(kp0_uv.cpu().numpy())
    # print(pos0_covTc)
    # print(pos1_covTc)

    
    rr.init("MACVO_PYTHON", spawn=True)
    rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN), static=True)
    rr.log("/world/camera/", rr.Pinhole(focal_length=frame0.fx, width=frame0.width, height=frame0.height, principal_point=(704/2, 704/2)))

    rr.set_time_sequence("frame_idx", 0)

    np_image_1 = frame0.imageL[0].permute(1, 2, 0).cpu().numpy()
    np_image_1 = (np_image_1 * 255).astype(np.uint8)

    rr.log("/world/camera/0/keypoints", rr.Points2D(kp0_uv.cpu().numpy()))
    rr.log("/world/camera/0/depth", rr.DepthImage(image=depth_0_cpu))
    rr.log("/world/camera/0/covariance", rr.DepthImage(image=cov_0_cpu))
    rr.log("/world/camera/0/left", rr.Image(image=np_image_1, color_model="BGR"))

    np_image_2 = frame1.imageL[0].permute(1, 2, 0).cpu().numpy()
    np_image_2 = (np_image_2 * 255).astype(np.uint8)
    rr.log("/world/camera/1/keypoints", rr.Points2D(kp1_uv.cpu().numpy()))
    rr.log("/world/camera/1/depth", rr.DepthImage(image=depth_1_cpu))
    rr.log("/world/camera/1/covariance", rr.DepthImage(image=cov_1_cpu))
    rr.log("/world/camera/1/left", rr.Image(image=np_image_2, color_model="BGR"))