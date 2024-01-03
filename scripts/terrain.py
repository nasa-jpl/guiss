import bpy
import bmesh
from mathutils import noise, Matrix
import numpy as np
import yaml


# Parameters for Create_Procedural_Mesh 

#PLANE_SIZE = 20 #20          # 20x20x0 mesh
#SUBDIVS = 2 #256            # number of cuts = resolution
zoffset = 0.5 #0.15           # elevation offset
noiseStrength = 0.5 #0.6     # Strength of noise 
noiseScale = 0.5 #0.4         # Scale of noise 
H = 0.8                 # fractal increment factor
lacunarity = 2.0         # gap   
octaves = 8              # noise frequencies

# a list of various noise functions with some default values
noise_functions = [
    lambda p: noise.cell(p),
    lambda p: noise.fractal(p, H, lacunarity, octaves),
    lambda p: noise.hetero_terrain(p, H, lacunarity, octaves, 
                zoffset, noise_basis="VORONOI_F2"),
    lambda p: noise.hybrid_multi_fractal(p, H, lacunarity, octaves, 1),
    lambda p: noise.multi_fractal(p, H, lacunarity, octaves),
    lambda p: noise.noise(p),
    lambda p: noise.ridged_multi_fractal(p, H, lacunarity, octaves, 1, 0),
    lambda p: noise.turbulence(p, octaves, True),
    lambda p: noise.variable_lacunarity(p, 1),
]


def reset_plane(mesh):
    for vertex in mesh.vertices:
        vertex.co.z = 0


def modify_terrain(mesh):        
    for vertex in mesh.vertices:
        sample_pos = vertex.co.xyz*noiseScale
        sample_pos.resize_3d()
        sample_pos.z = zoffset

        noise_in_height = noise_functions[2](sample_pos)
        vertex.co.z += noise_in_height*noiseStrength
    print("Mesh modified")


def create_Procedural_Mesh(config, plane_name='Landscape'):
    
    ### Use randomness to create a procedural mesh from noise
    # Here we just generate the noisy terrain and then save a new blend file such that we can load later through terrain_mode=1
    # No other changes are made in the blend file

    blend_file_name = config['Terrain']['blend_file']
    load_blend_file("blend_files/"+blend_file_name+".blend")

    myPlane = bpy.data.objects[plane_name]
    myPlane.select_set(True)

    mesh = myPlane.data

    modify_terrain(mesh)
    
    for f in mesh.polygons:
        f.use_smooth = True
    
    save_blend_filename = config['Terrain']['save_blend_filename']
    bpy.ops.wm.save_as_mainfile(filepath="blend_files/"+save_blend_filename+".blend")

    # save the corresponding yaml
    yaml_info = {"Sites": {"camera_locs": [[0,0,3]]} }
    with open("configs/" + save_blend_filename+'.yaml', 'w') as file:
        yaml.dump(yaml_info, file)


def load_blend_file(myBlendPath):
    # open ___.blend file
    try:
        bpy.ops.wm.open_mainfile(filepath=myBlendPath)
        print("File opened!")
    except FileNotFoundError:
        print("ERROR: File not found!")



def rename_object(name):
    # Called right after an object is added
    # bpy.context.selected_objects should contain a single object
    for i, obj in enumerate(bpy.context.selected_objects):
        #print("Selected objects name: ", i, obj.name)
        if len(bpy.context.selected_objects) == 1:
            obj.name = name
        else:
            raise Exception("More than one objects selected in context!")
    return obj


def add_mesh(model_path, model_texture_path, rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
               trans_vec=(0, 0, 0), scale=1, name=None, bsdf_value_dict=None):
    # Import
    if model_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=model_path)
    else:
        raise NotImplementedError("Importing model of this type")

    obj = rename_object(name)

    # Compute world matrix
    trans_4x4 = Matrix.Translation(trans_vec)
    rot_4x4 = Matrix(rot_mat).to_4x4()
    scale_4x4 = Matrix(np.eye(4)) # don't scale here
    obj.matrix_world = trans_4x4 @ rot_4x4 @ scale_4x4
    # Scale
    obj.scale = (scale, scale, scale)



##### Rock Distribution #####

def create_rock_distribution(Rock_Density=0.4, Scale_Max = 0.45):
    rockNode = bpy.data.node_groups["RockGeometryNode"]
    rockNode.nodes["Distribute Points on Faces"].inputs["Density Factor"].default_value = Rock_Density
    rockNode.nodes["scaleNode"].inputs[3].default_value = Scale_Max

