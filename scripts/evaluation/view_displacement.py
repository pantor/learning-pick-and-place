import argparse
import copy

import cv2
import numpy as np
import tkinter
import PIL.Image, PIL.ImageTk

from actions.action import RobotPose
from data.loader import Loader
from utils.image import get_area_of_interest_new



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure displacement in image.')
    parser.add_argument('-c', '--collection', dest='collection', type=str, default='placing-eval')
    parser.add_argument('-e', '--episode', dest='episode', type=str, required=True)
    parser.add_argument('-ca', '--camera', dest='camera', type=str, default='ed')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    action, place_after, place_goal = Loader.get_action(args.collection, args.episode, 1, [args.camera + '-after', args.camera + '-goal'])
    new_pose = RobotPose(action.pose)

    if not args.save:
        def update_image():
            after_area = get_area_of_interest_new(place_after, action.pose, size_cropped=(200, 200))
            goal_area = get_area_of_interest_new(place_goal, new_pose, size_cropped=(200, 200))

            ontop = np.zeros((200, 200, 3), dtype=np.uint8)

            if len(after_area.mat.shape) == 3:
                after_area.mat = after_area.mat[:, :, 0]
                goal_area.mat = goal_area.mat[:, :, 0]

            ontop[:, :, 0] += (after_area.mat / 255).astype(np.uint8)
            ontop[:, :, 1] += (after_area.mat / 255).astype(np.uint8)
            ontop[:, :, 2] += (goal_area.mat / 255).astype(np.uint8)

            dx = action.pose.x - new_pose.x
            dy = action.pose.y - new_pose.y

            dx_new = np.abs(dx * np.cos(action.pose.a) + dy * np.sin(action.pose.a))
            dy_new = np.abs(-dx * np.sin(action.pose.a) + dy * np.cos(action.pose.a))

            dt = np.sqrt(dx_new**2 + dy_new**2)
            da = np.abs(action.pose.a - new_pose.a)

            print('---')
            print(f'X: {dx_new * 1e3:0.4f} [mm]')
            print(f'Y: {dy_new * 1e3:0.4f} [mm]')
            print(f'Translation: {dt * 1e3:0.4f} [mm]')
            print(f'Rotation: {da * 180/np.pi:0.4f} [deg]')

            im = PIL.Image.fromarray(ontop)
            imgtk = PIL.ImageTk.PhotoImage(image=im)
            return imgtk

        root = tkinter.Tk()

        img = update_image()
        lbl = tkinter.Label(root, image=img)
        lbl.pack()

        dt_step = 0.0005
        da_step = 0.02

        def callback():
            img = update_image()
            lbl.configure(image=img)
            lbl.image = img

        def callback_px():
            new_pose.x += dt_step * np.cos(action.pose.a) # [m]
            new_pose.y += dt_step * np.sin(action.pose.a) # [m]
            callback()

        def callback_mx():
            new_pose.x -= dt_step * np.cos(action.pose.a) # [m]
            new_pose.y -= dt_step * np.sin(action.pose.a) # [m]
            callback()

        def callback_py():
            new_pose.x -= dt_step * np.sin(action.pose.a) # [m]
            new_pose.y += dt_step * np.cos(action.pose.a) # [m]
            callback()

        def callback_my():
            new_pose.x += dt_step * np.sin(action.pose.a) # [m]
            new_pose.y -= dt_step * np.cos(action.pose.a) # [m]
            callback()

        def callback_pa():
            new_pose.a += da_step # [m]
            callback()

        def callback_ma():
            new_pose.a -= da_step # [m]
            callback()

        button_px = tkinter.Button(root, text="+x", command=callback_px)
        button_mx = tkinter.Button(root, text="-x", command=callback_mx)
        button_py = tkinter.Button(root, text="+y", command=callback_py)
        button_my = tkinter.Button(root, text="-y", command=callback_my)
        button_pa = tkinter.Button(root, text="+a", command=callback_pa)
        button_ma = tkinter.Button(root, text="-a", command=callback_ma)

        button_px.pack()
        button_mx.pack()
        button_py.pack()
        button_my.pack()
        button_pa.pack()
        button_ma.pack()

        root.mainloop()
