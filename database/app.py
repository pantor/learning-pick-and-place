from collections import defaultdict
import io
import json

import cv2
import flask
import flask_socketio
import numpy as np
import pymongo as pm
from werkzeug.datastructures import ImmutableOrderedMultiDict

from actions.action import Action
from config import Config
from data.loader import Loader
from utils.image import draw_pose, draw_around_box


class MyRequest(flask.Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(flask.Flask):
    request_class = MyRequest


app = MyFlask(__name__)
socketio = flask_socketio.SocketIO(app)


database = pm.MongoClient()['robot-learning']


@app.route('/api/collection-list')
def api_collection_list():
    collection_list = sorted(database.collection_names())
    return flask.jsonify(collection_list)


def filterFromRequestValues(values):
    result = defaultdict(dict)

    ops_mongo = {
        '=': '$eq',
        '<': '$lt',
        '>': '$gt',
        '!=': '$ne',
        '<=': '$lte',
        '>=': '$gte',
    }

    if values.get('query') and values.get('query', type=str) != '':
        for part in values.get('query', type=str).strip(', ').split(' '):
            if part.strip() == '':
                continue

            # Own parsing using key=value arguments
            if any(o in part for o in ops_mongo):
                op = next(o for o in ops_mongo if (o in part))
                key, val = part.split(op)

                if key[0].isdigit():
                    action_id, key = key.split('.')
                else:
                    action_id = 0

                if key in ['r', 'reward']:
                    result[f'actions.{action_id}.reward'][ops_mongo[op]] = int(val)
                elif key in ['d', 'd_final', 'final_d']:
                    result[f'actions.{action_id}.final_pose.d'][ops_mongo[op]] = float(val)
                elif key in ['s', 'suffix']:
                    result[f'actions.{action_id}.images.{val}']['$exists'] = True
                else:
                    result[key][ops_mongo[op]] = val
            else:
                result['id']['$regex'] = part

    if values.get('id'):
        result['id'] = {'$regex': values.get('id')}

    if values.get('suffix'):
        result[f'actions.0.images.{values.get("suffix", type=str)}'] = {'$exists': True}

    if values.get('reward') and values.get('reward', type=float) > -1:
        result['actions.0.reward'] = values.get('reward', type=float)

    if values.get('final_d_lower') and (values.get('final_d_lower', type=float) > 0.0 or values.get('final_d_upper', type=float) < 0.1):
        result['actions.0.final_pose.d'] = {'$gt': values.get('final_d_lower', type=float), '$lt': values.get('final_d_upper', type=float)}

    return result


@app.route('/api/<collection_name>')
@app.route('/api/<collection_name>/episodes')
def api_episodes(collection_name: str):
    if collection_name not in database.collection_names():
        return flask.abort(404)

    collection = database[collection_name]

    filter_dict = filterFromRequestValues(flask.request.values)

    agr = [
        {'$match': filter_dict},
        {'$project': {'_id': 0, 'id': 1, 'actions.type': 1, 'actions.reward': 1}},
        {'$facet': {
            'stats': [{'$count': 'length'}],
            'data': [
                {'$sort': {'id': -1}},
                {'$skip': flask.request.values.get('skip', 0, type=int)},
                {'$limit': flask.request.values.get('limit', 1e9, type=int)},
            ],
        }},
    ]
    result = list(collection.aggregate(agr))[0]
    return flask.jsonify({
        'episodes': result['data'],
        'stats': result['stats'][0] if result['stats'] else {},
    })

    filter_dict = filterFromRequestValues(flask.request.values)

    episodes = list(collection.find(
        filter_dict,
        {'_id': 0, 'id': 1, 'actions.reward': 1, 'actions.type': 1}
    ))

    return flask.jsonify({
        'episodes': episodes,
        'stats': {'length': len(episodes)},
    })


@app.route('/api/<collection_name>/actions')
def api_actions(collection_name: str):
    if collection_name not in database.collection_names():
        return flask.abort(404)

    collection = database[collection_name]

    filter_dict = filterFromRequestValues(flask.request.values)

    agr = [
        {'$match': filter_dict},
        {'$project': {'id': 1, 'actions': 1, 'length': {'$size': '$actions'}}},
        {'$unwind': {'path': '$actions', 'includeArrayIndex': 'action_id'}},
        {'$project': {'_id': 0, 'episode_id': '$id', 'type': '$actions.type', 'reward': '$actions.reward', 'action_id': 1, 'length': 1}},
        {'$facet': {
            'stats': [{'$count': 'length'}],
            'data': [
                {'$sort': {'episode_id': -1, 'action_id': 1}},
                {'$skip': flask.request.values.get('skip', 0, type=int)},
                {'$limit': flask.request.values.get('limit', 1e9, type=int)},
            ],
        }},
    ]
    result = list(collection.aggregate(agr))[0]
    return flask.jsonify({
        'actions': result['data'],
        'stats': result['stats'][0] if result['stats'] else {},
    })


@app.route('/api/<collection_name>/stats')
def api_stats(collection_name: str):
    if collection_name not in database.collection_names():
        return flask.abort(404)

    collection = database[collection_name]

    filter_dict = filterFromRequestValues(flask.request.values)

    agr = [
        {'$match': filter_dict},
        {'$facet': {
            'episodes': [{'$count': 'total'}],
            'actions': [
                {'$unwind': '$actions'},
                {'$count': 'total'},
            ],
            'grasp': [
                {'$unwind': '$actions'},
                {'$match': {'actions.type': 'grasp'}},
                {'$group': {'_id': None, 'average_reward': {'$avg': '$actions.reward'}, 'total': {'$sum': 1}}},
            ],
        }},
    ]
    result = list(collection.aggregate(agr))[0]
    return flask.jsonify({
        'total': collection.count(),  # Without query
        'episodes': result['episodes'][0] if result['episodes'] else 0,
        'actions': result['actions'][0] if result['actions'] else 0,
        'grasp': result['grasp'][0] if result['grasp'] else 0,
    })


@app.route('/api/<collection_name>/<episode_id>')
def api_episode(collection_name: str, episode_id: str):
    if collection_name not in database.collection_names():
        return flask.abort(404)

    collection = database[collection_name]
    episode = collection.find_one({'id': episode_id}, {'_id': 0})
    if not episode:
        app.logger.warn(f'Could not find episode {episode_id}')
        return flask.abort(404)

    return flask.jsonify(episode)


# @app.route('/api/<collection_name>/<episode_id>/<action_id>')
# def api_action(collection_name: str, episode_id: str, action_id: int):
#     if collection_name not in database.collection_names():
#         return flask.abort(404)
#     collection = database[collection_name]
#     episode = collection.find_one({'id': episode_id}, {'_id': 0})
#     if not episode:
#         app.logger.warn(f'Could not find episode {episode_id}')
#         return flask.abort(404)

#     return flask.jsonify(episode)


@app.route('/api/image/<collection_name>/<episode_id>/<action_id>/<suffix>')
def api_image(collection_name: str, episode_id: str, action_id: str, suffix: str):
    def send_image(image):
        _, image_encoded = cv2.imencode('.jpg', image)
        return flask.send_file(io.BytesIO(image_encoded), mimetype='image/jpeg')

    def send_empty_image():
        empty = np.zeros((480, 752, 1))
        cv2.putText(empty, '?', (310, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, 100, thickness=6)
        return send_image(empty)


    if flask.request.values.get('pose'):
        action = Action(data=json.loads(flask.request.values.get('pose')))
        image = Loader.get_image(collection_name, episode_id, int(action_id), suffix, images=action.images)
    else:
        try:
            action, image = Loader.get_action(collection_name, episode_id, int(action_id), suffix)
        except Exception:
            app.logger.warn('Could not find image:', collection_name, episode_id, action_id, suffix)
            return send_empty_image()

    if suffix not in action.images.keys():
        app.logger.warn(f'Could not find suffix {collection_name}-{episode_id}-{action_id}-{suffix}')
        return send_empty_image()

    draw_pose(image, action.pose, convert_to_rgb=True)
    # draw_pose(image, action.pose, convert_to_rgb=True, reference_pose=action.images[suffix]['pose'])

    if flask.request.values.get('box', default=0, type=int):
        draw_around_box(image, box=Config.box, draw_lines=True)

    return send_image(image.mat / 255)


@app.route('/api/upload-image', methods=['POST'])
def api_upload_image():
    collection = flask.request.values.get('collection')
    episode_id = flask.request.values.get('episode_id')
    action_id = flask.request.values.get('action_id', type=int)
    suffix = flask.request.values.get('suffix')
    filepath = Loader.get_image_path(collection, episode_id, action_id, suffix, image_format='png')
    filepath.parent.mkdir(exist_ok=True, parents=True)

    image_data = flask.request.data
    if flask.request.files:
        image_data = flask.request.files['file'].read()

    image_buffer = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(filepath), image)
    return flask.jsonify(success=True)


@app.route('/api/new-episode', methods=['POST'])
def api_new_episode():
    collection_name = flask.request.values.get('collection')
    collection = database[collection_name]
    data = json.loads(flask.request.values.get('json'))['episode']
    data['collection'] = collection_name
    socketio.emit('new-episode', data)
    collection.insert_one(data)
    return flask.jsonify(success=True)


@app.route('/api/new-attempt', methods=['POST'])
def api_new_attempt():
    data = json.loads(flask.request.values.get('json'))
    socketio.emit('new-attempt', data)
    return flask.jsonify(success=True)


@app.route('/api/<collection_name>/<episode_id>/<action_id>/update-reward', methods=['POST'])
def api_update_reward(collection_name: str, episode_id: str, action_id: str):
    collection = database[collection_name]
    new_reward = flask.request.json['reward']
    if new_reward is None:
        return flask.jsonify(success=False)

    update_result = collection.update_one({'id': episode_id}, {'$set': { f'actions.{int(action_id)}.reward': new_reward }})
    return flask.jsonify(success=(update_result.matched_count == 1))


@app.route('/api/<collection_name>/<episode_id>/delete', methods=['POST'])
def api_delete(collection_name: str, episode_id: str):
    collection = database[collection_name]
    collection.delete_one({'id': episode_id})
    return flask.jsonify(success=True)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return flask.render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, use_reloader=False)
