import cv2
import numpy as np
import imageio

from PIL import Image
from habitat.utils.visualizations import maps


def move_to(sim, position=[0.0, 0.0, 0.0], angle=270):
    position = sim.pathfinder.snap_point(np.array(position))
    print("position: {}".format(position))
    
    angle = np.deg2rad(angle)
    print("angle: {}".format(np.rad2deg(angle)))

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = position
    state.rotation = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
    agent.set_state(state)

    return sim.get_sensor_observations()

def get_top_down_map(sim, position=[0.0, 0.0, 0.0], angle=270):
    position = sim.pathfinder.snap_point(np.array(position))
    print("position: {}".format(position))
    
    angle = np.deg2rad(angle)
    print("angle: {}".format(np.rad2deg(angle)))

    tdm = maps.get_topdown_map_from_sim(sim)
    pos = maps.to_grid(position[2], position[0], tdm.shape[:2], sim)
    tdm = maps.colorize_topdown_map(tdm)
    tdm = maps.draw_agent(tdm, pos, angle, agent_radius_px=20)
    return tdm

def add_label(img, text):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.25
    txt_size = cv2.getTextSize(text, font_face, scale, 1)

    margin = 8
    txt_width, txt_height = txt_size[0]
    img_height, img_width = img.shape[:2]
    pos = ((img_width - txt_width) // 2, txt_height + margin)

    color = (255, 255, 255)
    cv2.rectangle(img, (0, 0), (img_width, txt_height + 2 * margin), color, cv2.FILLED)

    color = (0, 0, 0)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def save_video(path, frames, label=None):
    writer = imageio.get_writer(path, fps=5, quality=5)
    for frame in frames:
        if label is not None:
            add_label(frame, label)
        writer.append_data(frame)
    writer.close()
    
def draw_path(top_down_map, path_points, color: int = 10, thickness: int = 2):
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            color,
            thickness=thickness,
            lineType=cv2.LINE_4,
        )
    
def make_transparent(img):
    img = Image.fromarray(img)
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return np.array(img)

def add_margin(pil_img, top, right, bottom, left, color=(0,0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def add_foreground(background, foreground, pad1=0, pad2=0):
    foreground = Image.fromarray(foreground)
    foreground = add_margin(foreground, pad1, 0, pad2, 0)
    foreground = foreground.resize(background.size)
    background.paste(foreground, (0, 0), foreground)
    return background