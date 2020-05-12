import cv2
import numpy as np

from data.loader import Loader
from utils.image import draw_around_box, get_inference_image, get_transformation


class TemplateMatcher:
    def __init__(self):
        pass

    def test(self, collection, episode_id):
        grasp = (Loader.get_image(collection, episode_id, 0, 'ed-v').mat / 255).astype(np.uint8)
        place = (Loader.get_image(collection, episode_id, 1, 'ed-v').mat / 255).astype(np.uint8)
        goal = (Loader.get_image(collection, episode_id, 0, 'ed-goal').mat / 255).astype(np.uint8)

        grasp_c = cv2.cvtColor(grasp, cv2.COLOR_GRAY2RGB)
        goal_c = cv2.cvtColor(goal, cv2.COLOR_GRAY2RGB)

        # Difference
        diff = cv2.absdiff(place, goal)
        diff[:80, :] = 0
        diff[-80:, :] = 0
        diff[:, :80] = 0
        diff[:, -80:] = 0

        # Find contours
        ret, thresh = cv2.threshold(diff, 20, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print('Number contours: ', len(contours))


        cv2.drawContours(goal_c, contours, -1, (255, 0, 0))

        # Bounding rect of largest area
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(goal_c, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Template matching
        template = goal[y:y+h, x:x+w]
        res = cv2.matchTemplate(grasp, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(grasp_c, top_left, bottom_right, (0, 0, 255), 1)

        cv2.imshow('grasp', grasp_c)
        cv2.imshow('goal', goal_c)
        cv2.waitKey(2000)


if __name__ == '__main__':

    selection_episodes = {
        '2020-02-12-11-26-00-515': 1,
        '2020-02-12-11-27-00-348': 1,
        '2020-02-12-11-28-37-251': 1,
        '2020-02-12-11-29-14-115': 0,
        '2020-02-12-11-32-45-449': 0,
        '2020-02-12-13-02-07-681': 1,
        '2020-02-12-13-02-43-881': 1,
        '2020-02-12-13-03-20-981': 1,
        '2020-02-12-13-04-00-948': 1,
        '2020-02-12-13-05-57-680': 1,
        '2020-02-12-13-07-08-581': 1,
        '2020-02-12-13-07-48-048': 1,
        '2020-02-12-13-08-31-048': 0,
        '2020-02-12-13-09-08-514': 0,
        '2020-02-12-13-09-45-347': 1,
        '2020-02-12-13-10-51-058': 0,
        '2020-02-12-13-11-29-980': 1,
        '2020-02-12-13-13-55-381': 1,
        '2020-02-12-13-14-35-781': 1,
        '2020-02-12-13-15-59-647': 1,
        '2020-02-12-13-16-42-581': 1,
        '2020-02-12-13-17-16-315': 1,
        '2020-02-12-13-19-40-025': 0,
        '2020-02-12-13-20-30-837': 1,
        '2020-02-12-13-21-02-115': 0,
        '2020-02-12-13-21-54-614': 0,
        '2020-02-12-13-22-33-314': 1,
        '2020-02-12-13-23-11-381': 1,
        '2020-02-12-13-26-18-217': 1,
        '2020-02-12-13-27-28-814': 1,
        '2020-02-12-13-30-02-380': 1,
        '2020-02-12-13-30-41-414': 1,
        '2020-02-12-13-31-23-448': 0,
        '2020-02-12-13-31-57-281': 1,
        '2020-02-12-13-33-17-715': 1,
        '2020-02-12-13-34-01-123': 1,
        '2020-02-12-13-34-59-613': 1,
        '2020-02-12-13-35-29-848': 1,
        '2020-02-12-13-36-06-780': 1,
        '2020-02-12-13-36-41-348': 0,
        '2020-02-12-13-37-15-846': 1,
        '2020-02-12-13-37-58-180': 1,
        '2020-02-12-13-40-42-614': 0,
        '2020-02-12-13-42-33-914': 1,
        '2020-02-12-13-43-10-514': 1,
        '2020-02-12-13-43-48-583': 0,
        '2020-02-12-13-44-14-679': 1,
    }

    episodes_list = list(selection_episodes.keys())

    ep = None
    for e in episodes_list:
        if selection_episodes[e] == -1:
            ep = e
            break

    if ep:
        print(f'Episode {ep}')

        tm = TemplateMatcher()
        tm.test(
            collection='placing-eval',
            episode_id=ep,
        )

    r = np.array([selection_episodes[k] for k in selection_episodes])
    print(f'Mean success: {r.mean():0.2f} ({r.std() / np.sqrt(len(r)):.3f})')
