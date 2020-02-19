from typing import Any, List

from data.loader import Loader


class GoalDatabase:
    single_peg = [
        ('placing-2', '2019-10-08-17-13-50-735', 0, ['ed-v']),
        ('placing-2', '2019-10-08-17-13-26-800', 0, ['ed-v']),
        ('placing-2', '2019-10-08-17-13-00-798', 0, ['ed-v']),
        ('placing-2', '2019-10-10-17-20-09-058', 0, ['ed-v']),
        ('placing-2', '2019-10-10-17-20-31-976', 0, ['ed-v']),
        ('placing-2', '2019-10-10-17-20-52-776', 0, ['ed-v']),
        ('placing-2', '2019-10-11-09-45-06-213', 0, ['ed-v']),
        ('placing-2', '2019-10-11-09-45-25-009', 0, ['ed-v']),
        ('placing-3', '2019-10-21-15-18-25-619', 0, ['ed-v']),
        ('placing-3', '2019-10-22-15-43-15-167', 0, ['ed-v']),
        ('placing-3', '2019-10-22-15-55-01-873', 0, ['ed-v']),
    ]

    single_cylinder = [
        ('placing-2', '2019-09-30-15-18-38-279', 1, ['ed-goal']),  # Middle
        ('placing-2', '2019-09-26-16-46-48-059', 0, ['ed-v']),  # Middle
        ('placing-2', '2019-10-07-13-17-54-572', 1, ['ed-goal']),  # Middle
        ('placing-2', '2019-09-26-16-52-09-226', 0, ['ed-v']),  # Middle / edge
        ('placing-2', '2019-09-26-16-48-37-325', 0, ['ed-v']),  # Middle / edge
        ('placing-2', '2019-09-26-17-15-28-259', 1, ['ed-goal']),  # Border
    ]

    two_cube = [
        ('placing-3', '2019-11-28-09-54-20-754', 0, ['ed-v']),  # 3 Tower
    ]

    two_peg = [
        ('placing-3', '2019-10-23-16-28-27-404', 0, ['ed-v']), 
    ]

    five_peg = [
        ('placing-3', '2019-10-17-17-13-23-472', 0, ['ed-v']),
        ('placing-3', '2019-11-25-13-52-03-671', 0, ['ed-v']),
        ('placing-3', '2019-11-25-13-52-29-058', 0, ['ed-v']),
        ('placing-3', '2019-11-25-13-57-30-588', 0, ['ed-v']),
        ('placing-3', '2019-11-25-16-24-52-163', 0, ['ed-v']),
        ('placing-3', '2019-11-25-17-41-37-723', 0, ['ed-v']),
        ('placing-3', '2019-11-25-17-16-12-813', 0, ['ed-v']),
        ('placing-3', '2019-11-25-20-36-21-691', 0, ['ed-v']),
        ('placing-3', '2019-11-25-20-34-20-559', 0, ['ed-v']),
        ('placing-3', '2019-11-25-20-30-20-637', 0, ['ed-v']),
        ('placing-3', '2019-11-25-20-21-07-144', 0, ['ed-v']),
        ('placing-3', '2019-11-26-17-41-51-644', 0, ['ed-goal']),
    ]

    many = [
        ('placing-3', '2019-11-28-16-44-21-179', 0, ['ed-v']),
    ]

    baby = [
        ('placing-3', '2020-01-22-14-30-35-856', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-14-22-43-155', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-14-21-29-189', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-14-18-48-360', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-14-18-48-360', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-14-15-48-519', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-14-03-40-251', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-13-57-12-490', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-22-13-50-42-555', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-49-37-425', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-46-18-721', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-42-04-417', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-39-45-465', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-29-52-785', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-22-13-15-55-116', 0, ['ed-v', 'rd-v', 'rc-v']),
        ('placing-3', '2020-01-24-11-45-27-205', 1, ['ed-after', 'rd-after', 'rc-after']),
        ('placing-3', '2020-01-27-12-35-24-959', 1, ['ed-after', 'rd-after', 'rc-after']),
    ]

    @classmethod
    def get_episodes_between(cls, collection: str, lower_id: str, upper_id: str = None, grasp_success=False, suffix=('ed-v',)):
        query = {'id': {'$gte': lower_id}}
        if upper_id:
            query['id']['$lte'] = upper_id

        if grasp_success:
            query['actions.0.reward'] = 1

        episodes = Loader.yield_episodes(collection, query=query)
        return list((d, e['id'], 0, suffix) for d, e in episodes)

    @classmethod
    def get(cls, config: str) -> List[Any]:
        if config == '1-cube':
            return cls.get_episodes_between('placing-3', '2019-10-15-20-26-52-766', '2019-10-16-10-41-22-351')
        if config == '1-peg':
            return cls.single_peg # + cls.get_episodes_between('placing-3', '2019-10-18-13-49-45-502', '2019-10-18-15-05-32-207')
        if config == '1-cylinder':
            return cls.single_cylinder
        if config == '1-cube-2-peg':
            return cls.get_episodes_between('placing-3', '2019-10-11-11-10-02-728', '2019-10-15-09-34-22-668')
        if config == '2-cube':
            return cls.two_cube
        if config == '2-peg':
            return cls.two_peg
        if config == '5-peg':
            return cls.five_peg
        if config == 'many':
            return cls.many
        if config == 'diff':
            return cls.get_episodes_between('placing-3', '2019-12-16-17-01-18-409', '2019-12-20-15-10-21-389', grasp_success=True)
        if config == 'screw':
            return cls.get_episodes_between('placing-screw-1', '2020-01-20-16-48-47-484', '2020-01-22-16-05-16-290', grasp_success=True, suffix=('ed-v', 'rd-v', 'rc-v'))
        if config == 'baby':
            return cls.baby
        return []
