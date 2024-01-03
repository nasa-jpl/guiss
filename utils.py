
import bpy
from mathutils import Matrix
import sys
import math
import numpy as np
from scripts.terrain import rename_object


def delete_objects(ignore_list=[]):
    for obj in bpy.data.objects:
        if obj.name in ignore_list:
            continue
        obj.select_set(True)
    bpy.ops.object.delete()


def get_rot_tuple(tuple_deg):
    # Convers tuple from degrees to radians
    return ( math.radians(tuple_deg[0]), 
             math.radians(tuple_deg[1]), 
             math.radians(tuple_deg[2]) )


#################################################
################## Lighting #####################

def update_lighting(config):
    light = bpy.data.lights["Light"]
    light.energy = config['energy'] # W/m^2, vs. 1361.0 W/m^2 on earth: 1/25
    light.angle = config['angular_diameter']

    light = bpy.data.objects["Light"]
    light.location = config["translation"]
    light.rotation_mode = 'XYZ'
    light.rotation_euler = get_rot_tuple(config['rotation'])


def add_light_source(light_type, config, name):
    # https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.light_add
    # light_type choices: "SPOT", "SUN", "SPOT", "AREA"
    
    bpy.ops.object.light_add(type=light_type)
    obj = rename_object(name)

    obj.rotation_mode = 'XYZ'
    obj.location = config['translation']
    obj.rotation_euler = get_rot_tuple(config['rotation'])

    obj.data.energy = config['energy'] # W/m^2, vs. 1361.0 W/m^2 on earth: 1/25
    obj.data.angle = config['angular_diameter']


#################################################
########### UPDATE STEREOSCOPIC CAMERAS #########

def update_stereoCam(baseline, focalLength, sensor_fit, sensor_width, sensor_height):
    stereoCam = bpy.data.cameras["StereoCamera"]
    stereoCam.stereo.interocular_distance = baseline # in meters
    stereoCam.lens = focalLength
    stereoCam.sensor_fit = sensor_fit
    stereoCam.sensor_width = sensor_width
    stereoCam.sensor_height = sensor_height
    stereoCam.clip_start = 0.01
    stereoCam.clip_end = 10000000.0
    #print(dir(stereoCam))



def add_manual_stereo_cam(options):
    cam_left = add_camera(name='CamLeft', 
                                f=options['camera_focal_length'],
                                sensor_width=options['sensor_width'],
                                sensor_height=options['sensor_height'],
                                sensor_fit=options['sensor_fit'])
    cam_right = add_camera(name='CamRight', 
                                 f=options['camera_focal_length'],
                                 sensor_width=options['sensor_width'],
                                 sensor_height=options['sensor_height'],
                                 sensor_fit=options['sensor_fit'])
    connectFrames(cam_left, cam_right, t_parent_child=(options['camera_baseline'], 0.0, 0.0), rot_parent_child=(0.0, 0.0, 0.0))

    return cam_left, cam_right



def add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None,
               proj_model='PERSP', f=35, sensor_fit='HORIZONTAL',
               sensor_width=60, sensor_height=18):
    #bpy.ops.object.camera_add(enter_editmode=False, align='VIEW')
    bpy.ops.object.camera_add()
    cam = bpy.data.objects['Camera']
    #cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.rotation_mode = 'XYZ'
    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000000.0
    return cam


# Set left camera as parent of right camera and establish stereo baseline
def connectFrames(parent_frame, child_frame, t_parent_child, rot_parent_child):
    child_frame.parent = parent_frame
    child_frame.rotation_mode = 'XYZ'
    child_frame.location = t_parent_child
    child_frame.rotation_euler = rot_parent_child

