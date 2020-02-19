from typing import Dict, List

import rospy  # pylint: disable=E0401

from bin_picking.srv import GetOrthographicImages  # pylint: disable=E0401, E0611

import utils.path_fix_ros  # pylint: disable=W0611

from cv_bridge import CvBridge  # pylint: disable=E0611

from orthographical import OrthographicImage


class Camera:
    def __init__(self, camera_suffixes: List[str]):
        self.suffixes = camera_suffixes
        self.bridge = CvBridge()

        self.ensenso_suffixes = [s for s in self.suffixes if s in ['ed', 'er']]
        self.realsense_suffixes = [s for s in self.suffixes if s in ['rd', 'rc']]

        if len(self.ensenso_suffixes) + len(self.realsense_suffixes) != len(self.suffixes):
            raise Exception('Unknown camera suffix in {self.suffixes}!')

        if self.ensenso_suffixes:
            rospy.wait_for_service('ensenso/images')
            self.ensenso_service = rospy.ServiceProxy('ensenso/images', GetOrthographicImages)

        if self.realsense_suffixes:
            rospy.wait_for_service('realsense/images')
            self.realsense_service = rospy.ServiceProxy('realsense/images', GetOrthographicImages)

    def take_images(self) -> List[OrthographicImage]:
        def add_camera(service, suffixes: List[str], images: Dict[str, OrthographicImage]) -> None:
            result = service(suffixes)
            for i, img in enumerate(result.images):
                mat = self.bridge.imgmsg_to_cv2(img.image, img.image.encoding)
                images[suffixes[i]] = OrthographicImage(mat, img.pixel_size, img.min_depth, img.max_depth, img.camera)

        images: Dict[str, OrthographicImage] = {}

        if self.ensenso_suffixes:
            add_camera(self.ensenso_service, self.ensenso_suffixes, images)

        if self.realsense_suffixes:
            add_camera(self.realsense_service, self.realsense_suffixes, images)
        return [images[s] for s in self.suffixes]
