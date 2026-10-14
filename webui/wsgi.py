"""Gunicorn entry point: load Kronos Small once per worker before serving."""

from webui.app import app


with app.test_client() as client:
    response = client.post('/api/load-model', json={'model_key': 'kronos-small', 'device': 'cpu'})
    if response.status_code != 200:
        raise RuntimeError(f'Kronos Small preload failed: {response.get_json()}')
