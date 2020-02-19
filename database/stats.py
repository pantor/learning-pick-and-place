from collections import defaultdict

import pymongo as pm

from data.loader import Loader

actions_on_date = defaultdict(lambda: 0)
episode_count = 0
action_count = 0
image_count = 0

database = pm.MongoClient()['robot-learning']

pipeline = [
    { '$project': {'id': 1, 'actions.images': 1}},
    { '$addFields': {
        'episode_helper': { '$divide': [1, {'$size': '$actions'}] },
    }},
    { '$unwind': '$actions' },
    { '$group': {
        '_id': { '$substr': [ "$id", 0, 7 ] },
        'episode_count': { '$sum': '$episode_helper' },
        'action_count': { '$sum': 1 },
        'image_count': { '$sum': { '$size': {'$objectToArray': '$actions.images'}}},
    }},
]


for c in Loader.get_collections():
    for e in list(database[c].aggregate(pipeline)):
        actions_on_date[e['_id']] += e['action_count']
        episode_count += int(e['episode_count'])
        action_count += e['action_count']
        image_count += e['image_count']


print(f'Total Episode Count: \t{episode_count}')
print(f'Total Action Count: \t{action_count}')
print(f'Total Image Count: \t{image_count}')
print()
for m in sorted(actions_on_date):
    print(f'{m}: {actions_on_date[m]}')
