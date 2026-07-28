# This is my second public python project, this is why I made for myself because I actually 
# needed a program to help with my blender renders, a huge part of this project has been getting it running as efficently as possible
# Its not the greatest looking thing but from my testing its worked reliably, as a disclaimer ai was used in this project, mostly as a debugger
# but the section ai created has been clearly labeled, all other work was done by me with the help of a range of youtube guides
# and the pyvista main page.




#git add .
#git commit -m "Fixed a giant issue with generation scaling"
#git push origin main
#cd OpenPLG
#git pull origin main
import tkinter as tk
from tkinter import filedialog
import os
import tempfile
import zipfile
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np
import pyvista as pv
from scipy.ndimage import zoom

detail = (125, 125)
scale = 0.5
initial_octaves = 6
vert_stretch = 1.0
val_min = -1.5
val_max = 1.5
use_ridges = False
use_biomes = False
use_render_mode = False
initial_x_offset = 0.0
initial_y_offset = 0.0
initial_water = -0.1
initial_weathering = 0
initial_export_res = 400
initial_thermal = 0
initial_hydro = 0
use_rivers = False

def apply_weathering(grid_map, passes):
    for _ in range(int(passes)):
        smoothed = np.copy(grid_map)
        smoothed[1:-1, 1:-1] = (
            grid_map[1:-1, 1:-1] * 0.6 +
            grid_map[:-2, 1:-1] * 0.1 +
            grid_map[2:, 1:-1] * 0.1 +
            grid_map[1:-1, :-2] * 0.1 +
            grid_map[1:-1, 2:] * 0.1
        )
        grid_map = smoothed
    return grid_map
def apply_advanced_simulations(grid_map, thermal_passes, hydro_passes, enable_rivers):
    for _ in range(int(thermal_passes)):
        smoothed = (np.roll(grid_map, 1, axis=0) + np.roll(grid_map, -1, axis=0) + 
                    np.roll(grid_map, 1, axis=1) + np.roll(grid_map, -1, axis=1)) / 4.0
        diff = grid_map - smoothed
        grid_map = np.where(diff > 0.04, grid_map - diff * 0.4, grid_map)
    for _ in range(int(hydro_passes)):
        gy, gx = np.gradient(grid_map)
        slope = np.sqrt(gx**2 + gy**2)
        carving = slope * 0.05
        deposition = np.roll(carving, 1, axis=0) * 0.02
        grid_map = grid_map - carving + deposition
    if enable_rivers:
        gy, gx = np.gradient(grid_map)
        flow_accumulation = np.sqrt(gx**2 + gy**2)
        river_channels = np.where(flow_accumulation > np.percentile(flow_accumulation, 85), 0.15, 0.0)
        grid_map -= river_channels
    return grid_map
#The following section was made with heavy help from an ai, I couldn't find the problem with it, and fix it myself
def generate_noise(octaves_val, scale_val, stretch_val, ridge_val, x_off, y_off, weathering_val, therm_val, hydro_val, river_val, grid_dims):
    #This makes sure if the grid is too big to downscale it so no crashy crashy.
    if grid_dims[0] > 200 or grid_dims[1] > 200:
        sample_dims = (150, 150)
        base_grid = generate_noise(octaves_val, scale_val, stretch_val, ridge_val, x_off, y_off, weathering_val, therm_val, hydro_val, river_val, sample_dims)
        zoom_factors = (grid_dims[0] / sample_dims[0], grid_dims[1] / sample_dims[1])
        return np.clip(zoom(base_grid, zoom_factors, order=1), val_min, val_max)
    #Makes to layers for better detail, if you wanted you could add more, wouldn't recommend it though
    base_noise = PerlinNoise(octaves=int(octaves_val))
    detail_noise = PerlinNoise(octaves=int(octaves_val * 2.5))
    grid_map = np.zeros(grid_dims)
    #Loops through setting the height for every point, this is the main visual thing.
    for i in range(grid_dims[0]):
        for j in range(grid_dims[1]):
            x = ((i / grid_dims[0]) + x_off) * scale_val
            y = ((j / grid_dims[1]) + y_off) * scale_val * stretch_val
            val = base_noise([x, y])
            val += 0.35 * detail_noise([x, y])
###############################################################################################3
            #Adds jagged sharp hills
            if ridge_val:
                val = (1.0 - abs(val)) ** 2
            grid_map[i][j] = val * stretch_val
            #Adds weather erosion
    grid_map = apply_weathering(grid_map, weathering_val)
    grid_map = apply_advanced_simulations(grid_map, therm_val, hydro_val, river_val)
    #Need this to clamp values because otherwise it looks insane
    return np.clip(grid_map, val_min, val_max)
grid = pv.RectilinearGrid(range(detail[0]), range(detail[1]), [0])
initial_terrain = generate_noise(initial_octaves, scale, vert_stretch, use_ridges, initial_x_offset, initial_y_offset, initial_weathering, initial_thermal, initial_hydro, use_rivers, detail)
grid.point_data["elevation"] = initial_terrain.ravel(order="F")
grid.point_data["geometry_elevation"] = np.copy(grid.point_data["elevation"])
grid.point_data["biomes"] = np.zeros_like(grid.point_data["elevation"])
mesh_3d = grid.warp_by_scalar("elevation", factor=50)
mesh_2d = grid.copy()
water_plane = pv.Plane(
    center=(detail[0] / 2, detail[1] / 2, initial_water * 50),
    direction=(0, 0, 1),
    i_size=detail[0],
    j_size=detail[1],
    i_resolution=1,
    j_resolution=1
)
plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 900])
plotter.enable_anti_aliasing('fxaa')
plotter.subplot(0, 0)
plotter.set_background(color="#181818")
plotter.add_title("2D TOPOGRAPHIC MAP", font_size=11, color="#AAAAAA")
actor_2d = plotter.add_mesh(mesh_2d, scalars="elevation", cmap="terrain", show_edges=False, show_scalar_bar=False)
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.subplot(0, 1)
plotter.set_background(color="#3D5A80", top="#0F172A")
plotter.add_title("3D TERRAIN SIMULATION", font_size=11, color="#AAAAAA")
plotter.enable_shadows()
actor_3d = plotter.add_mesh(
    mesh_3d, 
    scalars="elevation", 
    cmap="terrain", 
    pbr=True, 
    metallic=0.05, 
    roughness=0.80, 
    smooth_shading=True,
    show_scalar_bar=False)
actor_water = plotter.add_mesh(
    water_plane, 
    color="#005B96", 
    opacity=0.98, 
    specular=0.9,
    specular_power=60,
    diffuse=0.5,
    ambient=0.3,
    show_scalar_bar=False
)
plotter.view_isometric()
current_state = {
    "octaves": initial_octaves, 
    "scale": scale, 
    "stretch": vert_stretch,
    "ridges": use_ridges,
    "biomes": use_biomes,
    "render_mode": use_render_mode,
    "x_off": initial_x_offset,
    "y_off": initial_y_offset,
    "water": initial_water,
    "weathering": initial_weathering,
    "export_res": initial_export_res,
    "thermal": initial_thermal,
    "hydro": initial_hydro,
    "rivers": use_rivers
}
def update_viewports():
    new_terrain = generate_noise(
        current_state["octaves"], 
        current_state["scale"], 
        current_state["stretch"],
        current_state["ridges"],
        current_state["x_off"],
        current_state["y_off"],
        current_state["weathering"],
        current_state["thermal"],
        current_state["hydro"],
        current_state["rivers"],
        detail
    )
    flat_terrain = new_terrain.ravel(order="F")
    grid.point_data["geometry_elevation"] = flat_terrain
    if current_state["biomes"]:
        water_z = current_state["water"]
        biome_data = np.zeros_like(flat_terrain)
        biome_data[flat_terrain < water_z + 0.05] = -1.0 
        biome_data[(flat_terrain >= water_z + 0.05) & (flat_terrain < 0.3)] = -0.2 
        biome_data[(flat_terrain >= 0.3) & (flat_terrain < 0.8)] = 0.5 
        biome_data[flat_terrain >= 0.8] = 1.5 
        display_scalars = biome_data
    else:
        display_scalars = flat_terrain
    grid.point_data["elevation"] = display_scalars
    actor_2d.mapper.dataset.point_data["elevation"] = display_scalars
    updated_3d = grid.warp_by_scalar("geometry_elevation", factor=50)
    actor_3d.mapper.dataset.points = updated_3d.points
    actor_3d.mapper.dataset.point_data["elevation"] = display_scalars
    water_z = current_state["water"] * 50
    water_plane.points[:, 2] = water_z
def export_mesh(state):
    if not state:
        return
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    save_path = filedialog.asksaveasfilename(
        title="Export Terrain Package",
        defaultextension=".zip",
        filetypes=[("ZIP Archive", "*.zip")]
    )
    if not save_path:
        root.destroy()
        return
    if current_state["render_mode"]:
        res = int(current_state["export_res"])
        print(f"Rendering high-resolution export ({res}x{res})...")
        exp_dims = (res, res)
        geo_matrix = generate_noise(
            current_state["octaves"], 
            current_state["scale"], 
            current_state["stretch"],
            current_state["ridges"],
            current_state["x_off"],
            current_state["y_off"],
            current_state["weathering"],
            current_state["thermal"],
            current_state["hydro"],
            current_state["rivers"],
            exp_dims
        )
        temp_grid = pv.RectilinearGrid(range(exp_dims[0]), range(exp_dims[1]), [0])
        temp_grid.point_data["elevation"] = geo_matrix.ravel(order="F")
        export_mesh_obj = temp_grid.warp_by_scalar("elevation", factor=50)
        geo_data = geo_matrix
    else:
        export_mesh_obj = actor_3d.mapper.dataset.copy()
        geo_data = grid.point_data["geometry_elevation"].reshape(detail, order="F")
    h_min, h_max = np.min(geo_data), np.max(geo_data)
    heightmap = (geo_data - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(geo_data)
    watermask = np.where(geo_data <= current_state["water"], 1.0, 0.0)
    gy, gx = np.gradient(geo_data)
    slopemap = np.sqrt(gx**2 + gy**2)
    s_max = np.max(slopemap)
    if s_max > 0:
        slopemap = slopemap / s_max
    snowmask = np.where((heightmap > 0.65) & (slopemap < 0.25), 1.0, 0.0)
    with tempfile.TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, "terrain.ply")
        export_mesh_obj.save(ply_path)
        hm_path = os.path.join(temp_dir, "heightmap.png")
        plt.imsave(hm_path, heightmap.T, cmap="gray", origin="lower")
        wm_path = os.path.join(temp_dir, "watermask.png")
        plt.imsave(wm_path, watermask.T, cmap="gray", origin="lower")
        sm_path = os.path.join(temp_dir, "slopemap.png")
        plt.imsave(sm_path, slopemap.T, cmap="gray", origin="lower")
        sn_path = os.path.join(temp_dir, "snowmask.png")
        plt.imsave(sn_path, snowmask.T, cmap="gray", origin="lower")
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(ply_path, arcname="terrain.ply")
            zipf.write(hm_path, arcname="textures/heightmap.png")
            zipf.write(wm_path, arcname="textures/watermask.png")
            zipf.write(sm_path, arcname="textures/slopemap.png")
            zipf.write(sn_path, arcname="textures/snowmask.png")
    print(f"Successfully exported terrain package to: {save_path}")
    root.destroy()
def on_octave_change(value): current_state["octaves"] = int(value); update_viewports()
def on_scale_change(value): current_state["scale"] = value; update_viewports()
def on_stretch_change(value): current_state["stretch"] = value; update_viewports()
def on_x_pan(value): current_state["x_off"] = value; update_viewports()
def on_y_pan(value): current_state["y_off"] = value; update_viewports()
def on_water_change(value): current_state["water"] = value; update_viewports()
def on_weathering_change(value): current_state["weathering"] = int(value); update_viewports()
def on_thermal_change(value): current_state["thermal"] = int(value); update_viewports()
def on_hydro_change(value): current_state["hydro"] = int(value); update_viewports()
def on_export_res_change(value): current_state["export_res"] = int(value)
def on_ridge_change(value): current_state["ridges"] = value; update_viewports()
def on_biome_change(value): current_state["biomes"] = value; update_viewports()
def on_river_change(value): current_state["rivers"] = value; update_viewports()
def on_render_mode_change(value): current_state["render_mode"] = value
plotter.subplot(0, 0)
plotter.add_slider_widget(callback=on_x_pan, rng=[-5.0, 5.0], value=initial_x_offset, title="Pan X", pointa=(0.05, 0.10), pointb=(0.28, 0.10), style="modern")
plotter.add_slider_widget(callback=on_y_pan, rng=[-5.0, 5.0], value=initial_y_offset, title="Pan Y", pointa=(0.35, 0.10), pointb=(0.58, 0.10), style="modern")
plotter.add_slider_widget(callback=on_export_res_change, rng=[100, 1000], value=initial_export_res, title="Export Res", pointa=(0.65, 0.10), pointb=(0.95, 0.10), style="modern")
plotter.add_checkbox_button_widget(callback=on_ridge_change, value=use_ridges, position=(25, 580), size=24, border_size=1, color_on="#2EC4B6", color_off="#4A4E69")
plotter.add_text("Ridged Noise", position=(60, 583), font_size=10, color="white")
plotter.add_checkbox_button_widget(callback=on_biome_change, value=use_biomes, position=(25, 540), size=24, border_size=1, color_on="#3A86EF", color_off="#4A4E69")
plotter.add_text("Biome Mask", position=(60, 543), font_size=10, color="white")

#Not currently working, if you want to have a go, feel free to try to get it working.
#plotter.add_checkbox_button_widget(callback=on_river_change, value=use_rivers, position=(25, 500), size=24, border_size=1, color_on="#00F5D4", color_off="#4A4E69")
#plotter.add_text("River Pathing", position=(60, 503), font_size=10, color="white")

#Not currently working, i might add this later
#plotter.add_checkbox_button_widget(callback=on_render_mode_change, value=use_render_mode, position=(25, 460), size=24, border_size=1, color_on="#FF9F1C", color_off="#4A4E69")
#plotter.add_text("High-Res Render Mode", position=(60, 463), font_size=10, color="white")
plotter.add_checkbox_button_widget(callback=export_mesh, value=False, position=(25, 410), size=24, border_size=1, color_on="#E71D36", color_off="#2B2D42")
plotter.add_text("Export Package", position=(60, 413), font_size=10, color="white")
plotter.subplot(0, 1)
plotter.add_slider_widget(callback=on_octave_change, rng=[1, 10], value=initial_octaves, title="Detail", pointa=(0.02, 0.18), pointb=(0.23, 0.18), style="modern")
plotter.add_slider_widget(callback=on_scale_change, rng=[0.1, 5.0], value=scale, title="Scale", pointa=(0.26, 0.18), pointb=(0.47, 0.18), style="modern")
plotter.add_slider_widget(callback=on_stretch_change, rng=[0.1, 3.0], value=vert_stretch, title="Y-Stretch", pointa=(0.50, 0.18), pointb=(0.71, 0.18), style="modern")
plotter.add_slider_widget(callback=on_water_change, rng=[-1.5, 1.5], value=initial_water, title="Water Lvl", pointa=(0.74, 0.18), pointb=(0.96, 0.18), style="modern")
plotter.add_slider_widget(callback=on_weathering_change, rng=[0, 20], value=initial_weathering, title="Weathering", pointa=(0.02, 0.07), pointb=(0.31, 0.07), style="modern")
plotter.add_slider_widget(callback=on_thermal_change, rng=[0, 20], value=initial_thermal, title="Thermal Ero", pointa=(0.35, 0.07), pointb=(0.64, 0.07), style="modern")
plotter.add_slider_widget(callback=on_hydro_change, rng=[0, 20], value=initial_hydro, title="Hydro Ero", pointa=(0.67, 0.07), pointb=(0.96, 0.07), style="modern")
plotter.show()
