
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

## Functions for sampling viewpoints around sampling sites
## Define different sampling paradigms. Examples:
##  - Robot is at the cam loc and poses are sampled around it
##  - Cam loc is the site location and we sample a viewsphere around it


def sample_panorama(init_cam_loc, n_headings=24, pitch_list=[0.6, 0.8, 1.0], roll_list=[0], added_height=0):
    # Samples poses for a panorama 
    # Assume cam_loc is robot camera and is stable at sample site origin looking outwards

    yaw_list = [0.0] + [math.pi * 2 / n_headings * i for i in range(1, n_headings)][::-1]
    cam_rots = np.array( np.meshgrid(pitch_list, roll_list, yaw_list) ).T.reshape(-1,3)

    init_cam_loc[2] += added_height
    cam_locs = np.expand_dims(np.asarray(init_cam_loc), axis=0)
    cam_locs = np.repeat(cam_locs, cam_rots.shape[0], axis=0)
    
    return cam_locs, cam_rots


# assume the camera was moved to a new location and you want to get the rotation that points at the center 	
def get_new_rot(cam_pos, origin, up):
	# calculate camera x,y,z directions
	look_at_vec = np.subtract(origin,cam_pos)
	#print look_at_vec
	z_new = look_at_vec
	x_new = np.cross(up, look_at_vec)
	y_new = np.cross(z_new, x_new)
	# normalize the vectors
	z_new = z_new / np.linalg.norm(z_new)
	x_new = x_new / np.linalg.norm(x_new)
	y_new = y_new / np.linalg.norm(y_new)
	# concatenate to R
	R_cam = np.zeros((3,3), dtype=np.float32)
	R_cam[0,:] = -x_new
	R_cam[1,:] = y_new
	R_cam[2,:] = -z_new
	return R_cam


# def rot2eul(R):
#     pitch = -np.arcsin(R[2,0])
#     roll = np.arctan2(R[2,1]/np.cos(pitch),R[2,2]/np.cos(pitch))
#     yaw = np.arctan2(R[1,0]/np.cos(pitch),R[0,0]/np.cos(pitch))
#     return pitch, roll, yaw #np.array((roll, pitch, yaw))


def sample_viewsphere(site_loc, azim_lim=(0,360), elev_lim=(90,0), radius_list=[1], step=30):
    # Sample viewsphere
    # Assume site_loc is the sampling site at the center of the sphere and the robot is at radius
    site_loc = np.asarray(site_loc)

    azimuth_list = [math.radians(x) for x in list(range(azim_lim[0],azim_lim[1],step))]
    if elev_lim[1]<elev_lim[0]:
        step=-step
    elevation_list = [math.radians(x) for x in list(range(elev_lim[0],elev_lim[1],step))]
    
    #n_poses = len(azimuth_list) * len(elevation_list) * len(radius_list)
    cam_locs = []
    cam_rots = []
    
    # Given azimuth, elevation, radius, sample points on a sphere
    for radius in radius_list:
        for elev in elevation_list:
            for azim in azimuth_list:
                #spherical sampling (converting cartesian spherical coordinates ISO convention in wikipedia page to model's coordinate system) # theta->elevation, phi->azimuth
                x = radius*math.cos(azim)*math.sin(elev) # x=r*sin(theta)*cos(phi) [ISO] --> x [model]
                y = radius*math.sin(azim)*math.sin(elev) # y=r*sin(theta)*sin(phi) [ISO] --> y [model]
                z = radius*math.cos(elev) 				 # z=r*cos(theta) [ISO] 		 --> z [model]

                camera_pos = np.asarray([x,y,z])
                #print("azim:",azim, "elev:", elev, "cam_pos:", camera_pos)

                new_R = get_new_rot(camera_pos, origin=[0,0,0], up=[0,0,1]) # get the rotation pointing to the origin
                
                #angles = rot2eul(new_R)
                rot = R.from_matrix(new_R)
                angles = rot.as_euler("zyx") # -yaw, roll, -pitch (render takes: pitch, roll, yaw)
                angles[0] = -angles[0]
                angles[2] = -angles[2]
                angles[0], angles[2] = angles[2], angles[0]
                #print(angles)

                # center camera_pos around the sampling site
                camera_pos += site_loc
                #print("Pos:", camera_pos, "Rot:", angles)
                cam_locs.append(camera_pos)
                cam_rots.append(angles)
    
    cam_locs = np.stack(cam_locs, axis=0) # n_poses * 3
    cam_rots = np.stack(cam_rots, axis=0)
    return cam_locs, cam_rots
