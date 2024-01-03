
import yaml
import os
import itertools
from itertools import permutations
from itertools import product



EXEC_CMD = "../blender_versions/blender-3.6.0-linux-x64/./blender "
SCRIPT_CMD = "--python icyMoon_sim.py -- "

# "scene_reconstructions" uses terrain_mode=2 and loads the original europa scenes
# "texture_variation" uses terrain_mode=1 and loads a specific blend file with different textures
#       need to categorize the textures into different levels (noisier vs smoother)?
# "gaea_texture_variation" same as texture_variation but uses only the three gaea meshes
# "generative_texture" uses terrain_mode=1 and loads a specific blend file with procedural texture and the albedo without texture file (operates on glacier material)
# "terrain_variation" uses terrain_mode=1 and loads all blend files with a specific texture
# "rocks" uses terrain_mode=1 and loads a specific blend file with varying rock formations over a few textures
# "generative_texture_snow" same as generative_texture but operates on the snow material and has subsurface capabilities


def tex_name_helper(filename):
    if filename is not None:
        tex_name = "AlbedoMaps/"+filename
    else:
        tex_name = None
    return tex_name


def save_func(yaml_info, dataset_type, name, comb_count, cmd_file, mode_cmd):
    # Store the yaml and the bash file
    save_dir = "configs/job_yamls/" + dataset_type + "/" + name + "/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(save_dir+name+"_"+str(comb_count)+'.yaml', 'w') as file:
        yaml.dump(yaml_info, file)

    # write the command for this combination into the job file
    yaml_cmd = "--main_yaml " + "job_yamls/" + dataset_type + "/" + name + "/" + name + "_" + str(comb_count) + '.yaml'

    with open(cmd_file, "a") as rsh:
        rsh.write(EXEC_CMD + SCRIPT_CMD + mode_cmd + yaml_cmd + "\n")


def create_texture_dict():
    # Organize all textures and their respective files in a dict

    mesh_texture_list = ["icy-textures/aaron-burden-2GwUjTNX1CE-unsplash.jpg", "icy-textures/aaron-burden-if9vJoHDQes-unsplash.jpg", 
                        "icy-textures/angele-kamp-g8IEMx8p_z8-unsplash.jpg", "icy-textures/bernd-dittrich-gGpe31OT6hQ-unsplash.jpg",
                        "icy-textures/maaike-vrijenhoek--j-xuE8rZrg-unsplash.jpg", "icy-textures/powder-snow-texture.jpg"]

    albedo_texture_list = [ "Ground_Snow_seylmwd_4K_surface_ms/seylmwd_4K_albedo.jpg", "Snow005_4K-PNG/Snow005_4K_Color.png",
                            "Snow009B_4K-PNG/Snow009B_4K_Color.png", "ice_0001_4k_mGF6v8/ice_0001_ao_4k.jpg", 
                            "Snow_Mixed_ud4lfh2r_4K_surface_ms/ud4lfh2r_4K_AO.jpg", "Snow_Pure_ueiqbiefw_4K_surface_ms/ueiqbiefw_4K_AO.jpg",
                            "Snow_Pure_uepnbikfw_4K_surface_ms/uepnbikfw_4K_AO.jpg", "Snow_Pure_ugkieimdy_4K_surface_ms/ugkieimdy_4K_AO.jpg" ]

    albedo_roughness_list = [ "Ground_Snow_seylmwd_4K_surface_ms/seylmwd_4K_Roughness.jpg", "Snow005_4K-PNG/Snow005_4K_Roughness.png",
                            "Snow009B_4K-PNG/Snow009B_4K_Roughness.png", "ice_0001_4k_mGF6v8/ice_0001_roughness_4k.jpg",
                            "Snow_Mixed_ud4lfh2r_4K_surface_ms/ud4lfh2r_4K_Roughness.jpg", "Snow_Pure_ueiqbiefw_4K_surface_ms/ueiqbiefw_4K_Roughness.jpg",
                            "Snow_Pure_uepnbikfw_4K_surface_ms/uepnbikfw_4K_Roughness.jpg", "Snow_Pure_ugkieimdy_4K_surface_ms/ugkieimdy_4K_Roughness.jpg" ]

    albedo_normal_list = [ "Ground_Snow_seylmwd_4K_surface_ms/seylmwd_4K_normal.jpg", "Snow005_4K-PNG/Snow005_4K_NormalGL.png",
                            "Snow009B_4K-PNG/Snow009B_4K_NormalGL.png", "ice_0001_4k_mGF6v8/ice_0001_normal_directx_4k.png",
                            "Snow_Mixed_ud4lfh2r_4K_surface_ms/ud4lfh2r_4K_Normal.jpg", "Snow_Pure_ueiqbiefw_4K_surface_ms/ueiqbiefw_4K_Normal.jpg",
                            "Snow_Pure_uepnbikfw_4K_surface_ms/uepnbikfw_4K_Normal.jpg", "Snow_Pure_ugkieimdy_4K_surface_ms/ugkieimdy_4K_Normal.jpg" ]

    albedo_transmission_list = [ "Ground_Snow_seylmwd_4K_surface_ms/seylmwd_4K_transmission.jpg", None,
                                None, None,
                                "Snow_Mixed_ud4lfh2r_4K_surface_ms/ud4lfh2r_4K_Transmission.jpg", "Snow_Pure_ueiqbiefw_4K_surface_ms/ueiqbiefw_4K_Transmission.jpg",
                                "Snow_Pure_uepnbikfw_4K_surface_ms/uepnbikfw_4K_Transmission.jpg", "Snow_Pure_ugkieimdy_4K_surface_ms/ugkieimdy_4K_Transmission.jpg" ]

    # we do not use displacement files as it looks like they change the geometry of the scene
    albedo_displacement_list = [None, None, None, None, None, None, None, None]

    # populate a texture dict with the texture names pointing to the right file values
    texture_dict = {}

    for mesh_texture in mesh_texture_list:
        tex_name = mesh_texture.split('.')[0]
        texture_dict[tex_name] = {"mesh_texture_file": mesh_texture, "mesh_roughness_file": None, "mesh_normal_file": None, 
                                    "mesh_transmission_file": None, "mesh_disp_file": None}


    for i in range(len(albedo_texture_list)):
        tex_name = albedo_texture_list[i].split("/")[0]

        albedo_tex = tex_name_helper(albedo_texture_list[i])
        albedo_rough = tex_name_helper(albedo_roughness_list[i])
        albedo_normal = tex_name_helper(albedo_normal_list[i])
        albedo_trans = tex_name_helper(albedo_transmission_list[i])
        albedo_disp = tex_name_helper(albedo_displacement_list[i])

        texture_dict[tex_name] = {"mesh_texture_file": albedo_tex, "mesh_roughness_file": albedo_rough, 
                                    "mesh_normal_file": albedo_normal, "mesh_transmission_file": albedo_trans, 
                                    "mesh_disp_file": albedo_disp}
    
    return texture_dict


def gen_combs(dataset_type, cmd_file):

    #############################
    ### SCENE RECONSTRUCTIONS ###
    #############################

    if dataset_type == "scene_reconstructions":

        mesh_name_list = ["Athabasca_surface_2", "42m_before_NE", "bloodfalls", "seaice1", "Matanuska_Glacier_Site_A1", "Matanuska_Glacier_Site_C1", "svalbard1"]

        roughness_list = [0.5]
        subsurface_list = [0.0]
        specular_list = [0.0, 0.5, 1.0]
        metallic_list = [0.0]
        transmission_list = [0.0, 0.25, 0.5]
        albedo_list = [ [0.8, 0.8, 0.8, 1.0] ]

        brdf_combs = list(product(roughness_list, subsurface_list, specular_list, metallic_list, transmission_list, albedo_list))

        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0], [30,0,0], [60,0,0], [0,30,0], [0,60,0] ] #, [30,30,0], [30,60,0], [60,60,0], [60,30,0] ]
        sunlight_energy = [50.26] #[4.140, 50.26]
        angular_diameter = [0.01]

        sunlight_combs = list(product(sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter))

        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for mesh_name in mesh_name_list:
            comb_count = 0

            mesh_texture_file = "obj-textures/"+ mesh_name + ".jpg"

            for brdf_op in brdf_combs:

                for sunlight_op in sunlight_combs:

                    yaml_info["Terrain"]["mesh_name"] = mesh_name
                    yaml_info["Textures"]["mesh_texture_file"] = mesh_texture_file
                    yaml_info["Textures"]["mesh_roughness_file"] = None
                    yaml_info["Textures"]["mesh_normal_file"] = None
                    yaml_info["Textures"]["mesh_transmission_file"] = None
                    yaml_info["Textures"]["mesh_disp_file"] = None

                    yaml_info["Textures"]["Roughness"] = brdf_op[0]
                    yaml_info["Textures"]["Subsurface_Fac"] = brdf_op[1]
                    yaml_info["Textures"]["Specular"] = brdf_op[2]
                    yaml_info["Textures"]["Metallic"] = brdf_op[3]
                    yaml_info["Textures"]["Transmission"] = brdf_op[4]
                    yaml_info["Textures"]["Albedo_RGB"] = brdf_op[5]

                    yaml_info["SunLight"]["translation"] = sunlight_op[0]
                    yaml_info["SunLight"]["rotation"] = sunlight_op[1]
                    yaml_info["SunLight"]["energy"] = sunlight_op[2]
                    yaml_info["SunLight"]["angular_diameter"] = sunlight_op[3]

                    yaml_info["outputIdentifier"] = str(comb_count)

                    save_func(yaml_info, dataset_type, mesh_name, comb_count, cmd_file, mode_cmd="--terrain_mode 2 ")

                    comb_count+=1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    #########################
    ### TEXTURE VARIATION ###
    #########################

    elif dataset_type=="texture_variation":
        
        blend_file_list = [ "terrain_Enceladus_low_1", "terrain_Enceladus_med_1", "terrain_Enceladus_high_1" ]

        texture_dict = create_texture_dict()

        texture_names = texture_dict.keys()
        
        roughness_list = [0.5]
        subsurface_list = [0.0]
        specular_list = [0.5]
        metallic_list = [0.0]
        transmission_list = [0.0]
        albedo_list = [ [0.8, 0.8, 0.8, 1.0] ]
        
        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0] ]
        sunlight_energy = [4.140] #[4.140, 50.26]
        angular_diameter = [0.01]

        texture_rotation_list = [ [0,0,0] ]
        texture_scale_list = [ [0,0,0] ]
        noise_factor_list = [0.1]
        noise_detail_list = [10.0]

        param_combs = list(product(texture_names, roughness_list, subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                            sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter))

        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }
        
        for blend_file_name in blend_file_list:
            comb_count = 0

            for param_op in param_combs:
                texture_name = param_op[0]

                yaml_info["Terrain"]["blend_file"] = blend_file_name
                yaml_info["Textures"]["mesh_texture_file"] = texture_dict[texture_name]["mesh_texture_file"]
                yaml_info["Textures"]["mesh_roughness_file"] = texture_dict[texture_name]["mesh_roughness_file"]
                yaml_info["Textures"]["mesh_normal_file"] = texture_dict[texture_name]["mesh_normal_file"]
                yaml_info["Textures"]["mesh_transmission_file"] = texture_dict[texture_name]["mesh_transmission_file"]
                yaml_info["Textures"]["mesh_disp_file"] = texture_dict[texture_name]["mesh_disp_file"]

                yaml_info["Textures"]["Roughness"] = param_op[1]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[2]
                yaml_info["Textures"]["Specular"] = param_op[3]
                yaml_info["Textures"]["Metallic"] = param_op[4]
                yaml_info["Textures"]["Transmission"] = param_op[5]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[6]

                yaml_info["SunLight"]["translation"] = param_op[7]
                yaml_info["SunLight"]["rotation"] = param_op[8]
                yaml_info["SunLight"]["energy"] = param_op[9]
                yaml_info["SunLight"]["angular_diameter"] = param_op[10]

                # * Not really used, the code expects these fields
                yaml_info["Textures"]["Texture_Rotation"] = texture_rotation_list[0]
                yaml_info["Textures"]["Texture_Scale"] = texture_scale_list[0]
                yaml_info["Textures"]["Noise_Factor"] = noise_factor_list[0]
                yaml_info["Textures"]["Noise_Detail"] = noise_detail_list[0]

                yaml_info["outputIdentifier"] = str(comb_count)

                save_func(yaml_info, dataset_type, blend_file_name, comb_count, cmd_file, mode_cmd="--terrain_mode 1 ")

                comb_count+=1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    #####################################
    ### GAEA SCENES TEXTURE VARIATION ###
    #####################################

    elif dataset_type=="gaea_texture_variation":

        mesh_name_list = ["gaea_ridgy_terrain", "gaea_rockyPerlin", "gaea_rockyTerrain"]

        texture_dict = create_texture_dict()
        texture_names = texture_dict.keys()

        roughness_list = [0.5]
        subsurface_list = [0.0]
        specular_list = [0.5]
        metallic_list = [0.0]
        transmission_list = [0.0]
        albedo_list = [ [0.8, 0.8, 0.8, 1.0] ]
        
        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0] ]
        sunlight_energy = [4.140, 50.26]
        angular_diameter = [0.01]

        param_combs = list(product(texture_names, roughness_list, subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                            sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter))


        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for mesh_name in mesh_name_list:
            comb_count = 0

            for param_op in param_combs:
                texture_name = param_op[0]

                yaml_info["Terrain"]["mesh_name"] = mesh_name
                yaml_info["Textures"]["mesh_texture_file"] = texture_dict[texture_name]["mesh_texture_file"]
                yaml_info["Textures"]["mesh_roughness_file"] = texture_dict[texture_name]["mesh_roughness_file"]
                yaml_info["Textures"]["mesh_normal_file"] = texture_dict[texture_name]["mesh_normal_file"]
                yaml_info["Textures"]["mesh_transmission_file"] = texture_dict[texture_name]["mesh_transmission_file"]
                yaml_info["Textures"]["mesh_disp_file"] = texture_dict[texture_name]["mesh_disp_file"]

                yaml_info["Textures"]["Roughness"] = param_op[1]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[2]
                yaml_info["Textures"]["Specular"] = param_op[3]
                yaml_info["Textures"]["Metallic"] = param_op[4]
                yaml_info["Textures"]["Transmission"] = param_op[5]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[6]

                yaml_info["SunLight"]["translation"] = param_op[7]
                yaml_info["SunLight"]["rotation"] = param_op[8]
                yaml_info["SunLight"]["energy"] = param_op[9]
                yaml_info["SunLight"]["angular_diameter"] = param_op[10]

                yaml_info["outputIdentifier"] = str(comb_count)

                save_func(yaml_info, dataset_type, mesh_name, comb_count, cmd_file, mode_cmd="--terrain_mode 2 ")

                comb_count+=1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")



    ##########################
    ### GENERATIVE TEXTURE ###
    ##########################

    elif dataset_type=="generative_texture":

        # use different noise parameters to generate texture

        blend_file_list = [ "terrain_Enceladus_low_1" ]

        texture_rotation_list = [ [0,0,0] ]
        texture_scale_list = [ [0,0,0] ]
        noise_factor_list = [0.0, 0.25, 0.5, 0.75]
        noise_detail_list = [10.0]

        roughness_list = [0.5]
        subsurface_list = [0.0] #[0.0, 0.5, 1.0]
        specular_list = [0, 0.5, 1.0]
        metallic_list = [0.0]
        transmission_list = [0.0]
        albedo_list = [ [0.2, 0.2, 0.2, 1.0], [0.4, 0.4, 0.4, 1.0], [0.6, 0.6, 0.6, 1.0], [0.8, 0.8, 0.8, 1.0], [1.0, 1.0, 1.0, 1.0] ]

        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0] ]
        sunlight_energy = [2.0, 6.0, 10.0]
        angular_diameter = [0.01]

        waterplane_rgb_list = [ [0.4, 0.5, 0.75, 1.0] ] # color of plane below surface that contributes to the subsurface effect

        param_combs = list(product(texture_rotation_list, texture_scale_list, noise_factor_list, noise_detail_list, roughness_list, 
                                subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                                sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter, waterplane_rgb_list))

        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for blend_file_name in blend_file_list:
            comb_count = 0

            for param_op in param_combs:            

                yaml_info["Terrain"]["blend_file"] = blend_file_name
                yaml_info["Textures"]["mesh_texture_file"] = None
                yaml_info["Textures"]["mesh_roughness_file"] = None
                yaml_info["Textures"]["mesh_normal_file"] = None
                yaml_info["Textures"]["mesh_transmission_file"] = None
                yaml_info["Textures"]["mesh_disp_file"] = None

                yaml_info["Textures"]["Texture_Rotation"] = param_op[0]
                yaml_info["Textures"]["Texture_Scale"] = param_op[1]
                yaml_info["Textures"]["Noise_Factor"] = param_op[2]
                yaml_info["Textures"]["Noise_Detail"] = param_op[3]

                yaml_info["Textures"]["Roughness"] = param_op[4]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[5]
                yaml_info["Textures"]["Specular"] = param_op[6]
                yaml_info["Textures"]["Metallic"] = param_op[7]
                yaml_info["Textures"]["Transmission"] = param_op[8]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[9]

                yaml_info["SunLight"]["translation"] = param_op[10]
                yaml_info["SunLight"]["rotation"] = param_op[11]
                yaml_info["SunLight"]["energy"] = param_op[12]
                yaml_info["SunLight"]["angular_diameter"] = param_op[13]

                yaml_info["Textures"]["WaterPlane_RGB"] = param_op[14]

                yaml_info["outputIdentifier"] = str(comb_count)
                
                save_func(yaml_info, dataset_type, blend_file_name, comb_count, cmd_file, mode_cmd="--terrain_mode 1 ")

                comb_count += 1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    #########################
    ### TERRAIN VARIATION ###
    #########################

    elif dataset_type=="terrain_variation":

        blend_file_list = [ "terrain_Enceladus_low_2", "terrain_Enceladus_low_3", "terrain_Enceladus_med_2", "terrain_Enceladus_med_3", 
                            "terrain_Enceladus_med_4", "terrain_Enceladus_high_2", "terrain_Enceladus_high_3", "terrain_Enceladus_veryhigh_1" ]


        texture_dict = create_texture_dict()

        # use only a small subset of textures
        texture_names = ["icy-textures/bernd-dittrich-gGpe31OT6hQ-unsplash", "icy-textures/powder-snow-texture", 
                         "Ground_Snow_seylmwd_4K_surface_ms", "Snow_Mixed_ud4lfh2r_4K_surface_ms"]

        texture_dict = {key: texture_dict[key] for key in texture_names}


        roughness_list = [0.5]
        subsurface_list = [0.0]
        specular_list = [0.5]
        metallic_list = [0.0]
        transmission_list = [0.0]
        albedo_list = [ [0.8, 0.8, 0.8, 1.0] ]
        
        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0] ]
        sunlight_energy = [4.140] #[4.140, 50.26]
        angular_diameter = [0.01]

        texture_rotation_list = [ [0,0,0] ]
        texture_scale_list = [ [0,0,0] ]
        noise_factor_list = [0.1]
        noise_detail_list = [10.0]

        param_combs = list(product(texture_names, roughness_list, subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                            sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter))


        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for blend_file_name in blend_file_list:
            comb_count = 0

            for param_op in param_combs:    
                texture_name = param_op[0]

                yaml_info["Terrain"]["blend_file"] = blend_file_name
                yaml_info["Textures"]["mesh_texture_file"] = texture_dict[texture_name]["mesh_texture_file"]
                yaml_info["Textures"]["mesh_roughness_file"] = texture_dict[texture_name]["mesh_roughness_file"]
                yaml_info["Textures"]["mesh_normal_file"] = texture_dict[texture_name]["mesh_normal_file"]
                yaml_info["Textures"]["mesh_transmission_file"] = texture_dict[texture_name]["mesh_transmission_file"]
                yaml_info["Textures"]["mesh_disp_file"] = texture_dict[texture_name]["mesh_disp_file"]

                yaml_info["Textures"]["Roughness"] = param_op[1]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[2]
                yaml_info["Textures"]["Specular"] = param_op[3]
                yaml_info["Textures"]["Metallic"] = param_op[4]
                yaml_info["Textures"]["Transmission"] = param_op[5]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[6]

                yaml_info["SunLight"]["translation"] = param_op[7]
                yaml_info["SunLight"]["rotation"] = param_op[8]
                yaml_info["SunLight"]["energy"] = param_op[9]
                yaml_info["SunLight"]["angular_diameter"] = param_op[10]

                # * Not really used, the code expects these fields
                yaml_info["Textures"]["Texture_Rotation"] = texture_rotation_list[0]
                yaml_info["Textures"]["Texture_Scale"] = texture_scale_list[0]
                yaml_info["Textures"]["Noise_Factor"] = noise_factor_list[0]
                yaml_info["Textures"]["Noise_Detail"] = noise_detail_list[0]

                yaml_info["outputIdentifier"] = str(comb_count)

                save_func(yaml_info, dataset_type, blend_file_name, comb_count, cmd_file, mode_cmd="--terrain_mode 1 ")

                comb_count += 1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    #############
    ### ROCKS ###
    #############

    elif dataset_type=="rocks":

        blend_file_list = ["terrain_Enceladus_low_1_rocks"]

        texture_dict = create_texture_dict()

        # use only a small subset of textures
        texture_names = ["icy-textures/bernd-dittrich-gGpe31OT6hQ-unsplash", "icy-textures/powder-snow-texture", 
                         "Snow009B_4K-PNG", "Snow_Pure_ugkieimdy_4K_surface_ms"]

        texture_dict = {key: texture_dict[key] for key in texture_names}

        rock_density_factor_list = [0.5, 0.8]
        rock_scale_max_list = [0.5, 0.8]

        roughness_list = [0.5]
        subsurface_list = [0.0]
        specular_list = [0.5]
        metallic_list = [0.0]
        transmission_list = [0.0]
        albedo_list = [ [0.8, 0.8, 0.8, 1.0] ]
        
        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [0,0,0], [30,0,0], [60,0,0], [0,30,0], [0,60,0] ]
        sunlight_energy = [4.140] #[4.140, 50.26]
        angular_diameter = [0.01]

        texture_rotation_list = [ [0,0,0] ]
        texture_scale_list = [ [0,0,0] ]
        noise_factor_list = [0.1]
        noise_detail_list = [10.0]

        param_combs = list(product(texture_names, roughness_list, subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                            sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter, rock_density_factor_list, rock_scale_max_list))

        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for blend_file_name in blend_file_list:
            comb_count = 0

            for param_op in param_combs:    
                texture_name = param_op[0]

                yaml_info["Terrain"]["blend_file"] = blend_file_name
                yaml_info["Textures"]["mesh_texture_file"] = texture_dict[texture_name]["mesh_texture_file"]
                yaml_info["Textures"]["mesh_roughness_file"] = texture_dict[texture_name]["mesh_roughness_file"]
                yaml_info["Textures"]["mesh_normal_file"] = texture_dict[texture_name]["mesh_normal_file"]
                yaml_info["Textures"]["mesh_transmission_file"] = texture_dict[texture_name]["mesh_transmission_file"]
                yaml_info["Textures"]["mesh_disp_file"] = texture_dict[texture_name]["mesh_disp_file"]

                yaml_info["Textures"]["Roughness"] = param_op[1]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[2]
                yaml_info["Textures"]["Specular"] = param_op[3]
                yaml_info["Textures"]["Metallic"] = param_op[4]
                yaml_info["Textures"]["Transmission"] = param_op[5]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[6]

                yaml_info["SunLight"]["translation"] = param_op[7]
                yaml_info["SunLight"]["rotation"] = param_op[8]
                yaml_info["SunLight"]["energy"] = param_op[9]
                yaml_info["SunLight"]["angular_diameter"] = param_op[10]

                # * Not really used, the code expects these fields
                yaml_info["Textures"]["Texture_Rotation"] = texture_rotation_list[0]
                yaml_info["Textures"]["Texture_Scale"] = texture_scale_list[0]
                yaml_info["Textures"]["Noise_Factor"] = noise_factor_list[0]
                yaml_info["Textures"]["Noise_Detail"] = noise_detail_list[0]

                yaml_info["Terrain"]["Rock_Density_Factor"] = param_op[11]
                yaml_info["Terrain"]["Rock_Scale_Max"] = param_op[12]

                yaml_info["outputIdentifier"] = str(comb_count)

                save_func(yaml_info, dataset_type, blend_file_name, comb_count, cmd_file, mode_cmd="--terrain_mode 1 ")

                comb_count += 1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    ###############################
    ### GENERATIVE TEXTURE SNOW ###
    ###############################
        
    elif dataset_type=="generative_texture_snow":

        blend_file_list = ["terrain_Enceladus_low_1_procedural_subsurf"]

        # important params:
        # Subsurface factor, albedo_rgb, transmission, noise_factor, noise_detail

        texture_rotation_list = [ [0,0,0] ]
        texture_scale_list = [ [0,0,0] ]
        noise_factor_list = [0.0, 1.0, 2.0]
        noise_detail_list = [0.0, 1.0, 2.0]

        roughness_list = [0.5]
        subsurface_list = [0.0, 1.0]
        specular_list = [0.5]
        metallic_list = [0.0]
        transmission_list = [0.0, 0.5]
        albedo_list = [ [0.4, 0.4, 0.4, 1.0], [0.8, 0.8, 0.8, 1.0] ]

        sunlight_translation = [ [0,0,10] ]
        sunlight_rotation = [ [-66,0,0] ]
        sunlight_energy = [10.0]
        angular_diameter = [0.01]

        waterplane_rgb_list = [ [0.4, 0.5, 0.75, 1.0] ] # color of plane below surface that contributes to the subsurface effect

        param_combs = list(product(texture_rotation_list, texture_scale_list, noise_factor_list, noise_detail_list, roughness_list, 
                                subsurface_list, specular_list, metallic_list, transmission_list, albedo_list,
                                sunlight_translation, sunlight_rotation, sunlight_energy, angular_diameter, waterplane_rgb_list))

        yaml_info = { "Terrain":{}, "Textures": {"folder":"images/"}, "SunLight": {}, "demo":False, "dataset_type":dataset_type }

        for blend_file_name in blend_file_list:
            comb_count = 0

            for param_op in param_combs:            

                yaml_info["Terrain"]["blend_file"] = blend_file_name
                yaml_info["Textures"]["mesh_texture_file"] = None
                yaml_info["Textures"]["mesh_roughness_file"] = None
                yaml_info["Textures"]["mesh_normal_file"] = None
                yaml_info["Textures"]["mesh_transmission_file"] = None
                yaml_info["Textures"]["mesh_disp_file"] = None

                yaml_info["Textures"]["Texture_Rotation"] = param_op[0]
                yaml_info["Textures"]["Texture_Scale"] = param_op[1]
                yaml_info["Textures"]["Noise_Factor"] = param_op[2]
                yaml_info["Textures"]["Noise_Detail"] = param_op[3]

                yaml_info["Textures"]["Roughness"] = param_op[4]
                yaml_info["Textures"]["Subsurface_Fac"] = param_op[5]
                yaml_info["Textures"]["Specular"] = param_op[6]
                yaml_info["Textures"]["Metallic"] = param_op[7]
                yaml_info["Textures"]["Transmission"] = param_op[8]
                yaml_info["Textures"]["Albedo_RGB"] = param_op[9]

                yaml_info["Textures"]["WaterPlane_RGB"] = param_op[14]

                yaml_info["SunLight"]["translation"] = param_op[10]
                yaml_info["SunLight"]["rotation"] = param_op[11]
                yaml_info["SunLight"]["energy"] = param_op[12]
                yaml_info["SunLight"]["angular_diameter"] = param_op[13]

                yaml_info["outputIdentifier"] = str(comb_count)
                
                save_func(yaml_info, dataset_type, blend_file_name, comb_count, cmd_file, mode_cmd="--terrain_mode 1 ")

                comb_count += 1

        with open(cmd_file, "a") as rsh:
            rsh.write("\n")


    else:
        raise Exception("Unkown dataset type !!!")



if __name__ == '__main__':

    # create job bash file
    cmd_file = "run_gen_datasets.sh"
    with open (cmd_file, 'w') as rsh:
        rsh.write("#!/bin/sh\n\n\n")

    gen_combs(dataset_type="scene_reconstructions", cmd_file=cmd_file)
    gen_combs(dataset_type="texture_variation", cmd_file=cmd_file)
    gen_combs(dataset_type="gaea_texture_variation", cmd_file=cmd_file)
    gen_combs(dataset_type="generative_texture", cmd_file=cmd_file)
    gen_combs(dataset_type="terrain_variation", cmd_file=cmd_file)
    gen_combs(dataset_type="rocks", cmd_file=cmd_file)
    gen_combs(dataset_type="generative_texture_snow", cmd_file=cmd_file)