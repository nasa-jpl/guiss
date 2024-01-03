import bpy
import mathutils
import math


def texture_fileName_helper(folder=None, fileName=None):
        if fileName is not None:
            return folder + fileName
        else:
            return None


def setTextureOnObj(obj, mesh_texture_filename, material_name, bsdf_value_dict=None):

    material = bpy.data.materials.new(name=material_name)
    
    material.use_nodes = True
    
    material_output = material.node_tree.nodes.get('Material Output')
    image_texture_node = material.node_tree.nodes.new('ShaderNodeTexImage')

    bpy.ops.image.open(filepath=mesh_texture_filename)

    texture_filename_no_path = mesh_texture_filename.split("/")[-1]
    image_texture = bpy.data.images[texture_filename_no_path]
    image_texture_node.image = image_texture


    # https://docs.blender.org/manual/en/2.93/render/shader_nodes/shader/principled.html
    principled_bsdf = material.node_tree.nodes.get('Principled BSDF')
    #for i, o in enumerate(principled_bsdf.inputs):
    #    print(i, o.name, o.default_value)

    # If bsdf values are provided then apply. Otherwise keep default
    if bsdf_value_dict is not None:
        for k in bsdf_value_dict.keys():
            principled_bsdf.inputs[k].default_value = bsdf_value_dict[k]

    # Link emission shader to material
    material.node_tree.links.new(principled_bsdf.inputs['Base Color'], image_texture_node.outputs['Color'])
    material.node_tree.links.new(material_output.inputs['Surface'], principled_bsdf.outputs['BSDF'])

    # Shade smooth
    for poly in obj.data.polygons:
        poly.use_smooth = True
    
    # Set active material to your new material
    obj.active_material = material



#################################################
################ UPDATE TEXTURE #################

def update_texture(params): 
    # Material input node
    glacier_material = bpy.data.materials["glacier_Material"]
    glacier_node_tree = glacier_material.node_tree
    glacier_material_inputNode = []

    if glacier_material is None:
        print("Error: No active texture found!")
    else:
        glacier_material_inputNode = glacier_node_tree.nodes["glacierNodeGroup"]
        
    
    glacier_material_inputNode.inputs[3].default_value = params['Albedo_RGB'] # Base Color in RGBA
    glacier_material_inputNode.inputs[0].default_value = params['Roughness'] # Roughness
    glacier_material_inputNode.inputs[4].default_value = params['Subsurface_Fac'] # Subsurface
    glacier_material_inputNode.inputs[5].default_value = params['Metallic'] # Metallic
    glacier_material_inputNode.inputs[6].default_value = params['Specular'] # Specular
    glacier_material_inputNode.inputs[7].default_value = params['Transmission'] # Transmission
    
    glacier_material_inputNode.inputs[1].default_value = mathutils.Euler([math.radians(deg) for deg in params['Texture_Rotation']], 'XYZ') # Rotation
    glacier_material_inputNode.inputs[2].default_value = mathutils.Vector(params['Texture_Scale']) # Scale
    glacier_material_inputNode.inputs[8].default_value = params['Noise_Factor'] # Noise Factor
    glacier_material_inputNode.inputs[9].default_value = params['Noise_Detail'] # Noise Detail

    mesh_texture_filename = params['mesh_texture_file'] # Texture/Albedo Maps
    mesh_roughness_filename = params['mesh_roughness_file'] # Roughness Map
    mesh_normal_filename = params['mesh_normal_file'] # Normal Map
    mesh_transmission_filename = params['mesh_transmission_file'] # Transmission Map
    mesh_disp_filename = params['mesh_disp_file'] # Displacement Map
    use_procedural_texture = params['use_procedural_texture'] # Procedural method

    
    # TEXTURE
    if mesh_texture_filename is not None:
        bpy.ops.image.open(filepath=mesh_texture_filename)
        texture_filename_no_path = mesh_texture_filename.split("/")[-1]
        image_texture = bpy.data.images[texture_filename_no_path]
        glacier_node_tree.nodes["textureNode"].image = image_texture
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Base Color"], 
                                    glacier_node_tree.nodes["textureNode"].outputs["Color"])
    else:
        glacier_node_tree.nodes["textureNode"].image = None
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Base Color"], 
                                    glacier_material_inputNode.outputs["procedural_Base_Color"])
    

     # ROUGHNESS
    if mesh_roughness_filename is not None:
        bpy.ops.image.open(filepath=mesh_roughness_filename)
        glacier_node_tree.nodes["roughnessNode"].image = bpy.data.images[mesh_roughness_filename.split("/")[-1]]
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Roughness"], 
                                    glacier_node_tree.nodes["roughnessNode"].outputs["Color"])
    else:
        glacier_node_tree.nodes["roughnessNode"].image = None
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Roughness"], 
                                    glacier_material_inputNode.outputs["BSDF_Roughness"])



    # NORMAL
    # use Procedural texture for normal and displacement if textures are not specified

    if mesh_normal_filename is not None:
        bpy.ops.image.open(filepath=mesh_normal_filename)
        glacier_node_tree.nodes["normalNode"].image = bpy.data.images[mesh_normal_filename.split("/")[-1]]
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Normal"], 
                                    glacier_node_tree.nodes["Normal Map"].outputs["Normal"])
    elif use_procedural_texture:
        glacier_node_tree.nodes["normalNode"].image = None
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Normal"], 
                                    glacier_material_inputNode.outputs["procedural_Normal"])
    
    else: 
        glacier_node_tree.nodes["displacementNode"].image = None


    # TRANSMISSION
    if mesh_transmission_filename is not None:
        bpy.ops.image.open(filepath=mesh_transmission_filename)
        glacier_node_tree.nodes["transmissionNode"].image = bpy.data.images[mesh_transmission_filename.split("/")[-1]]
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Transmission"], 
                                    glacier_node_tree.nodes["transmissionNode"].outputs["Color"])
    else:
        glacier_node_tree.nodes["transmissionNode"].image = None
        glacier_node_tree.links.new(glacier_node_tree.nodes["Principled BSDF"].inputs["Transmission"], 
                                    glacier_material_inputNode.outputs["BSDF_Transmission"])
    

    # DISPLACEMENT
    if mesh_disp_filename is not None:
        bpy.ops.image.open(filepath=mesh_disp_filename)
        glacier_node_tree.nodes["displacementNode"].image = bpy.data.images[mesh_disp_filename.split("/")[-1]]
        glacier_node_tree.links.new(glacier_node_tree.nodes["Material Output"].inputs["Displacement"], 
                                    glacier_node_tree.nodes["Displacement"].outputs["Displacement"])
    elif use_procedural_texture:
        glacier_node_tree.nodes["displacementNode"].image = None
        glacier_node_tree.links.new(glacier_node_tree.nodes["Material Output"].inputs["Displacement"], 
                                    glacier_node_tree.nodes["iceTextDispNode"].outputs["Displacement"])



    #####################################################
    # Snow Material added -> procedural only
    if use_procedural_texture:
        snow_material = "snow_texture"
        
        if snow_material in bpy.data.materials:
            snow_material = bpy.data.materials[snow_material]
            print("Snow material found!")

            # Noise Texture Scale
            bpy.data.materials['snow_texture'].node_tree.nodes["Noise Texture"].inputs[2].default_value = params['Noise_Detail'] 
            # Noise Texture Distortion
            bpy.data.materials['snow_texture'].node_tree.nodes["Noise Texture"].inputs[5].default_value = params['Noise_Factor']
            # Subsurface Factor
            bpy.data.materials["snow_texture"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = params['Subsurface_Fac']
            # bpy.data.materials["snow_texture"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = params['Subsurface_RGB'] # Subsurf Color in RGBA 

            # transmission, albedo, albedo_plane
            bpy.data.materials["snow_texture"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = params['Albedo_RGB'] # Base Color in RGBA
            bpy.data.materials["snow_texture"].node_tree.nodes["Principled BSDF"].inputs[17].default_value = params['Transmission']

        water_material = "water_Material" # Water Plane 
        
        if water_material in bpy.data.materials:
            water_material = bpy.data.materials[water_material]
            print("Water material found!")

            bpy.data.materials["water_Material"].node_tree.nodes["Principled BSDF"].inputs[0].default_value = params['WaterPlane_RGB']



    # DEBUG:  Print the texture update parameters
    for input in glacier_material_inputNode.inputs:
        print("\n", f"Input {input.identifier} is named {input.name} with value of {input.default_value}")
