

#######################################
#######################################
# "Copyright 2023, by the California Institute of Technology. ALL RIGHTS RESERVED. 
# United States Government Sponsorship acknowledged. 
# Any commercial use must be negotiated with the Office of Technology 
# Transfer at the California Institute of Technology.
 
# This software may be subject to U.S. export control laws. 
# By accepting this software, the user agrees to comply with all applicable U.S. 
# export laws and regulations. User has the responsibility to obtain export licenses, 
# or other export authority as may be required before exporting such information to 
# foreign countries or providing access to foreign persons."
#######################################
#######################################


#######################################
## Graphical Utility for Icy Moon Surface Simulations (GUISS)
## Authors: Ram Bhaskara, Georgios Georgakis
#######################################


import os, signal
import sys
# Assuming this is ran in the root of the europa_sim repo. 
# Add current directory in path
BASE_PATH = os.getcwd() + "/"
sys.path.append(BASE_PATH)

import argparse
import bpy
import yaml
import json
import mathutils
import numpy as np
import cv2
import math
import utils
import pose_sampling as ps
import scripts.texture as tex
import scripts.terrain as terr


def set_cycles(samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    cycles = scene.cycles
    # easiest way to remove grainy noise is to increase number of samples
    cycles.samples = samples
    cycles.use_adaptive_sampling = True

    cycles.use_progressive_refine = True
    #if n_samples is not None:
    #    cycles.samples = n_samples
    cycles.max_bounces = 100
    cycles.min_bounces = 10
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5

    # Ensure no background node
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass


def set_scene(options):
    scene = bpy.data.scenes['Scene']
    scene.view_settings.exposure = options['exposure']
    #scene.display_settings.display_device = 'sRGB'
    #scene.view_settings.view_transform = 'Filmic'

    # World Lighting parameters
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = options['ambient_occlusion']
    world.light_settings.ao_factor = options['ambient_occlusion_factor']

    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    bpy.context.scene.view_layers[0].use_pass_z = True

    scene.render.resolution_x = options['width']
    scene.render.resolution_y = options['height']
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'

	# added code
    #bpy.context.scene.render.use_compositing = True
    #bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')      
    vl = tree.nodes.new('CompositorNodeViewer')   
    vl.use_alpha = False
    links.new(rl.outputs[2], vl.inputs[0]) # link Z to output


########################################
############### RENDER #################

def render_depth(filepath):
    # save the depth data
    z = bpy.data.images['Viewer Node']
    w, h = z.size
    dmap = np.array(z.pixels[:], dtype=np.float32)
    dmap = np.reshape(dmap, (h, w, 4))[:,:,0]
    dmap = np.rot90(dmap, k=2)
    dmap = np.fliplr(dmap)
    dmap[dmap>1000] = 0
    dist_from_cam = dmap[int(h/2), int(w/2)]
    #print(filepath, 'distance from cam', dist_from_cam)

    # Uncomment to normalize and save for visualization
    # max_depth = np.amax(dmap)
    # dmap_vis = dmap/max_depth * 255.0
    # cv2.imwrite(filepath + '_depth.png', dmap_vis)
    
    with open(filepath + ".npy", "wb") as f:
        np.save(filepath + ".npy", dmap)	
    f.close()


def render_to_file(cam, filepath, depth):
    # Set up default camera render
    bpy.context.scene.camera = cam
    # Render
    bpy.context.scene.render.filepath = filepath + ".png"
    #bpy.ops.render.render(use_viewport=True, write_still=True)
    bpy.ops.render.render(write_still=True)
    if depth:
        render_depth(filepath)


def render(store_dir, img_id=0, cam_loc=[0,0,0], cam_rot=[0,0,0], stereo=True, depth=True):
    # Move cameras...moving cam_left also moves cam_right 
    bpy.data.objects['CamLeft'].location = cam_loc
    bpy.data.objects['CamLeft'].rotation_euler = cam_rot

    img_path = store_dir + "/view_" + "%04i" % img_id
    render_to_file(cam=bpy.data.objects['CamLeft'], filepath=img_path+'_left', depth=depth)
    render_to_file(cam=bpy.data.objects['CamRight'], filepath=img_path+'_right', depth=False)


def set_environment(terrain_mode, config, options):

    # The light properties change given europa/enceladus
    light_config = config['SunLight']

    if terrain_mode == 1:
        # Case when blend file is provided
        blend_file_name = config['Terrain']['blend_file']
        terr.load_blend_file(BASE_PATH + "blend_files/"+blend_file_name+".blend")

        # Delete current cameras and deselect the stereoscopy option
        bpy.data.objects['Camera'].select_set(True)
        bpy.ops.object.delete()
        for c in bpy.data.cameras:
            bpy.data.cameras.remove(c)
        bpy.context.scene.render.use_multiview = False 

        # Ignore existing stereo cam and manually add our own (to match the other terrain modes)
        cam_left, cam_right = utils.add_manual_stereo_cam(options)

        #print("Objects:", bpy.data.objects.keys())
        #print("Cameras:", bpy.data.cameras.keys())

        # Light already exists in the blend file so we just update it
        utils.update_lighting(light_config)

        blend_file_config = yaml.safe_load(open(BASE_PATH + 'configs/'+blend_file_name+'.yaml', 'r'))

        # get sampling sites
        sampling_sites_locs = np.asarray(blend_file_config['Sites']['camera_locs'])

        if blend_file_name == "terrain_Enceladus_low_1_rocks":
            ###### Rock Distribution ########
            terr.create_rock_distribution(config['Terrain']['Rock_Density_Factor'],
                                        config['Terrain']['Rock_Scale_Max']
                                      )   
        
    
    elif terrain_mode == 2:
        # Case when obj is provided
        # Each mesh should come with its own yaml file
        utils.delete_objects() # Remove all default objects

        mesh_name = config['Terrain']['mesh_name']
        mesh_config = yaml.safe_load(open(BASE_PATH + 'configs/'+mesh_name+'.yaml', 'r'))
        
        # load the obj
        path_to_model = BASE_PATH + "meshes/"+mesh_name+"/"
        model_path = path_to_model + mesh_config['Mesh']['mesh_file'] + ".obj"
        terr.add_mesh(model_path=model_path, 
                       model_texture_path=None, # texture is handled later
                       rot_mat=mesh_config['Mesh']['rotation'],
                       trans_vec=mesh_config['Mesh']['translation'],
                       scale=mesh_config['Mesh']['scale'],
                       name=mesh_name)
        # Add stereo cam / light
        cam_left, cam_right = utils.add_manual_stereo_cam(options)

        utils.add_light_source(light_type="SUN", config=light_config, name="SunLight")

        sampling_sites_locs = np.asarray(mesh_config['Sites']['camera_locs'])
            
    else:
        raise Exception("Invalid terrain option!!!")

    return sampling_sites_locs



def set_texture(terrain_mode, config):

    texture_params = config['Textures']

    if terrain_mode == 1:

        texture_params['use_procedural_texture'] = False
        if texture_params['mesh_texture_file'] is None:
            texture_params['use_procedural_texture'] = True
        
        #base_path = "/home/georgakis/europa_sim/"

        # update the file paths with the entire path
        texture_params['mesh_texture_file'] = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_texture_file'])
        texture_params['mesh_roughness_file'] = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_roughness_file'])
        texture_params['mesh_normal_file'] = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_normal_file'])
        texture_params['mesh_transmission_file'] = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_transmission_file'])
        texture_params['mesh_disp_file'] = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_disp_file'])

        tex.update_texture(texture_params)
    
    elif terrain_mode == 2:

        model_texture_path = tex.texture_fileName_helper(BASE_PATH + texture_params['folder'], texture_params['mesh_texture_file'])

        bsdf_value_dict = {'Base Color': texture_params['Albedo_RGB'],
                           'Roughness': texture_params['Roughness'],
                           'Subsurface': texture_params['Subsurface_Fac'],
                           'Metallic': texture_params['Metallic'],
                           'Specular': texture_params['Specular'],
                           'Transmission': texture_params['Transmission']}

        obj = bpy.data.objects[config['Terrain']['mesh_name']]

        tex.setTextureOnObj(obj, model_texture_path, "MeshTexture", bsdf_value_dict)

    else:
        raise Exception("Invalid terrain option!!!")



if __name__ == '__main__':
    
    # Terrain modes 
    # 1: uses the scene in the provided blend file
    # 2: loads an external obj file
    # 3: procedurally generates a terrain
    # See README for more info

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--terrain_mode', type=int, default=1, choices=[1, 2, 3], help='Terrain geometry options. See icyMoon yaml for more details.')
    parser.add_argument('--main_yaml', type=str, default='icyMoon.yaml')
    parser.add_argument('--keep_blender_running', default=False, action='store_true', help="If enabled, kills blender process after rendering." )


    if '--' not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_args(argv)


    # Load the yaml with the common options
    options = yaml.safe_load(open(BASE_PATH + 'configs/common.yaml', 'r'))
    # Load the yaml with the simulation configuration options
    config = yaml.safe_load(open(BASE_PATH + 'configs/'+args.main_yaml, 'r'))


    # Setting rendering engine and general Blender options
    set_cycles(samples=options['cycles_samples'])
    set_scene(options)


    if args.terrain_mode == 3:
        # if terrain_mode=3 we procedurally generate the terrain and save the resulting blend file
        # Currently relevant params are in terrain_Enceladus.py
        terr.create_Procedural_Mesh(config)
    
    else:

        # Sets the terrain, cameras, and lighting based on config choices
        sampling_sites_locs = set_environment(args.terrain_mode, config, options)

        # Set texture depending on the terrain mode
        set_texture(args.terrain_mode, config)
        
        
        if config['demo']:
            store_dir = "visualizations/" + config["outputIdentifier"]
            if not os.path.isdir(store_dir):
                os.makedirs(store_dir)
            cam_loc = sampling_sites_locs[0,:]
            print("Cam loc:", cam_loc)
            cam_rot = [1.22173, 0, -1.74533] # degrees: [70,0,-100]
            #cam_rot = [0,0,0] # points downwards
            render(store_dir, 0, cam_loc, cam_rot=cam_rot, depth=True)

        else:
            if args.terrain_mode==1:
                main_store_dir = "data/dataset/"+config['dataset_type']+"/"+config['Terrain']['blend_file']+"/"+config['Terrain']['blend_file']+"_"+config["outputIdentifier"]
            elif args.terrain_mode==2:
                main_store_dir = "data/dataset/"+config['dataset_type']+"/"+config['Terrain']['mesh_name']+"/"+config['Terrain']['mesh_name']+"_"+config["outputIdentifier"]
            
            print("Saving in:", main_store_dir)

            if not os.path.isdir(main_store_dir):
                os.makedirs(main_store_dir)
                folder_id = 0
            else:
                # count the existing number of folders and re-assign the id
                data_dirs = os.listdir(main_store_dir)
                folder_id = len(data_dirs)


            for k in range(len(sampling_sites_locs)):
                cam_loc = sampling_sites_locs[k,:]
                print("########### Site:", k, cam_loc)

                store_dir = main_store_dir + "/site_" + str(k) + "_panorama_stereo"

                cam_locs, cam_rots = ps.sample_panorama(init_cam_loc=cam_loc, n_headings=4, pitch_list=[0.8], added_height=0)

                for i in range(cam_locs.shape[0]):
                    render(store_dir, img_id=i, cam_loc=cam_locs[i,:], cam_rot=cam_rots[i,:], depth=True)
    
                # store a json file with the relevant params used in this run
                config.update(options)
                with open(store_dir+"/params.json", "w") as outfile:
                    json.dump(config, outfile, indent=4)


    print("Finished!")

    if not args.keep_blender_running:
        # quit when done
        bpy.ops.wm.quit_blender() 

        # Manually kill blender process in case the cmd above does not work
        # This is needed when we are generating a dataset using the run_gen_datasets.sh (product of gen_yamls.py)
        # If blender is not killed, then the next command in the shell script does not initiate
        
        # iterating through each instance of the process
        for line in os.popen("ps ax | grep blender | grep -v grep"):
            fields = line.split()
                
            # extracting Process ID from the output
            pid = fields[0]
                
            # terminating process
            os.kill(int(pid), signal.SIGKILL)

