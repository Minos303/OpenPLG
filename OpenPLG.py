from perlin_noise import PerlinNoise
import numpy as np
import pyvista as pv

detail = (125, 125)
scale = 0.5
initial_octaves = 6
vert_stretch = 1.0
val_min = -1.5
val_max = 1.5
use_ridges = False
initial_x_offset = 0.0
initial_y_offset = 0.0
initial_water = -0.1
initial_weathering = 0

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

def generate_noise(octaves_val, scale_val, stretch_val, ridge_val, x_off, y_off, weathering_val):
    base_noise = PerlinNoise(octaves=int(octaves_val))
    detail_noise = PerlinNoise(octaves=int(octaves_val * 2.5))
    grid_map = np.zeros(detail)
    
    for i in range(detail[0]):
        for j in range(detail[1]):
            x = ((i / detail[0]) + x_off) * scale_val
            y = ((j / detail[1]) + y_off) * scale_val * stretch_val

            val = base_noise([x, y])
            val += 0.35 * detail_noise([x, y])

            if ridge_val:
                val = (1.0 - abs(val)) ** 2
                
            grid_map[i][j] = val * stretch_val

    grid_map = apply_weathering(grid_map, weathering_val)
    
    return np.clip(grid_map, val_min, val_max)

grid = pv.RectilinearGrid(range(detail[0]), range(detail[1]), [0])
initial_terrain = generate_noise(initial_octaves, scale, vert_stretch, use_ridges, initial_x_offset, initial_y_offset, initial_weathering)
grid.point_data["elevation"] = initial_terrain.ravel(order="F")

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

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
plotter.add_title("2D Noise Map", font_size=12)
actor_2d = plotter.add_mesh(mesh_2d, scalars="elevation", cmap="gist_earth", show_edges=False, show_scalar_bar=False)
plotter.view_xy()
plotter.enable_parallel_projection()

plotter.subplot(0, 1)
plotter.set_background(color="lightskyblue", top="midnightblue")
plotter.add_title("3D Terrain View", font_size=12)
actor_3d = plotter.add_mesh(
    mesh_3d, 
    scalars="elevation", 
    cmap="gist_earth", 
    pbr=True, 
    metallic=0.15, 
    roughness=0.6, 
    smooth_shading=True,
    show_scalar_bar=False
)
actor_water = plotter.add_mesh(water_plane, color="deepskyblue", opacity=0.7, pbr=True, roughness=0.1, metallic=0.2, show_scalar_bar=False)
plotter.view_isometric()

current_state = {
    "octaves": initial_octaves, 
    "scale": scale, 
    "stretch": vert_stretch,
    "ridges": use_ridges,
    "x_off": initial_x_offset,
    "y_off": initial_y_offset,
    "water": initial_water,
    "weathering": initial_weathering
}

def update_viewports():
    new_terrain = generate_noise(
        current_state["octaves"], 
        current_state["scale"], 
        current_state["stretch"],
        current_state["ridges"],
        current_state["x_off"],
        current_state["y_off"],
        current_state["weathering"]
    )
    
    flat_terrain = new_terrain.ravel(order="F")
    grid.point_data["elevation"] = flat_terrain
    
    actor_2d.mapper.dataset.point_data["elevation"] = flat_terrain
    
    updated_3d = grid.warp_by_scalar("elevation", factor=50)
    
    actor_3d.mapper.dataset.points = updated_3d.points
    actor_3d.mapper.dataset.point_data["elevation"] = flat_terrain
    
    water_z = current_state["water"] * 50
    water_plane.points[:, 2] = water_z

def on_octave_change(value):
    current_state["octaves"] = int(value)
    update_viewports()

def on_scale_change(value):
    current_state["scale"] = value
    update_viewports()

def on_stretch_change(value):
    current_state["stretch"] = value
    update_viewports()

def on_x_pan(value):
    current_state["x_off"] = value
    update_viewports()

def on_y_pan(value):
    current_state["y_off"] = value
    update_viewports()

def on_water_change(value):
    current_state["water"] = value
    update_viewports()

def on_weathering_change(value):
    current_state["weathering"] = int(value)
    update_viewports()

def on_ridge_change(value):
    current_state["ridges"] = value
    update_viewports()

plotter.subplot(0, 0)
plotter.add_slider_widget(callback=on_x_pan, rng=[-5.0, 5.0], value=initial_x_offset, title="Pan X", pointa=(0.15, 0.28), pointb=(0.85, 0.28), style="modern")
plotter.add_slider_widget(callback=on_y_pan, rng=[-5.0, 5.0], value=initial_y_offset, title="Pan Y", pointa=(0.15, 0.15), pointb=(0.85, 0.15), style="modern")

plotter.subplot(0, 1)
plotter.add_slider_widget(callback=on_octave_change, rng=[1, 10], value=initial_octaves, title="Detail", pointa=(0.02, 0.28), pointb=(0.32, 0.28), style="modern")
plotter.add_slider_widget(callback=on_scale_change, rng=[0.1, 5.0], value=scale, title="Scale", pointa=(0.35, 0.28), pointb=(0.65, 0.28), style="modern")
plotter.add_slider_widget(callback=on_stretch_change, rng=[0.1, 3.0], value=vert_stretch, title="Y-Stretch", pointa=(0.68, 0.28), pointb=(0.98, 0.28), style="modern")

plotter.add_slider_widget(callback=on_weathering_change, rng=[0, 20], value=initial_weathering, title="Weathering", pointa=(0.02, 0.15), pointb=(0.48, 0.15), style="modern")
plotter.add_slider_widget(callback=on_water_change, rng=[-1.5, 1.5], value=initial_water, title="Water Lvl", pointa=(0.52, 0.15), pointb=(0.98, 0.15), style="modern")

plotter.add_checkbox_button_widget(callback=on_ridge_change, value=use_ridges, position=(0.02, 0.88), size=30, border_size=2, color_on="green", color_off="red")

plotter.show()
