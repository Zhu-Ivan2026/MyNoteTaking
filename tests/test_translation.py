import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flask import Flask

from src.routes import translation


class TranslationRouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(translation.translation_bp, url_prefix='/api')
        self.client = app.test_client()
        self.prompt_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
        self.prompt_file.write('Custom translation instructions')
        self.prompt_file.close()
        self.prompt_path = Path(self.prompt_file.name)
        self.prompt_patcher = patch.object(translation, 'PROMPT_PATH', self.prompt_path)
        self.prompt_patcher.start()
        self.env_patcher = patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'})
        self.env_patcher.start()

    def tearDown(self):
        self.prompt_patcher.stop()
        self.env_patcher.stop()
        self.prompt_path.unlink(missing_ok=True)

    def test_rejects_missing_text_and_unsupported_language(self):
        with patch.object(translation.requests, 'post') as post:
            missing_text = self.client.post('/api/translate', json={'target_language': 'ja'})
            bad_language = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'xx'})
            japanese = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'ja'})
            invalid_language_type = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': []})

        self.assertEqual(missing_text.status_code, 400)
        self.assertEqual(bad_language.status_code, 400)
        self.assertEqual(japanese.status_code, 400)
        self.assertEqual(invalid_language_type.status_code, 400)
        post.assert_not_called()

    def test_reports_missing_api_key(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}), patch.object(translation.requests, 'post') as post:
            response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-CN'})

        self.assertEqual(response.status_code, 503)
        post.assert_not_called()

    def test_uses_editable_prompt_and_returns_translation(self):
        upstream = Mock()
        upstream.json.return_value = {'choices': [{'message': {'content': 'こんにちは'}}]}
        with patch.object(translation.requests, 'post', return_value=upstream) as post:
            response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-CN'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'translated_text': 'こんにちは'})
        call_payload = post.call_args.kwargs['json']
        self.assertEqual(call_payload['messages'][0]['content'], 'Custom translation instructions')
        self.assertIn('Simplified Chinese', call_payload['messages'][1]['content'])
        self.assertIn('Hello', call_payload['messages'][1]['content'])
        self.assertEqual(post.call_args.kwargs['timeout'], (5, 45))

    def test_reports_missing_prompt_file(self):
        self.prompt_path.unlink()
        response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-CN'})

        self.assertEqual(response.status_code, 500)
        self.assertIn('prompt', response.get_json()['error'].lower())

    def test_reports_upstream_timeout(self):
        with patch.object(translation.requests, 'post', side_effect=translation.requests.Timeout):
            response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-CN'})

        self.assertEqual(response.status_code, 504)

    def test_reports_upstream_error_without_provider_details(self):
        with patch.object(translation.requests, 'post', side_effect=translation.requests.ConnectionError('private detail')):
            response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-TW'})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn('private detail', response.get_json()['error'])

    def test_explains_openrouter_billing_limit(self):
        upstream_response = Mock(status_code=402)
        error = translation.requests.HTTPError(response=upstream_response)
        with patch.object(translation.requests, 'post', side_effect=error):
            response = self.client.post('/api/translate', json={'text': 'Hello', 'target_language': 'zh-CN'})

        self.assertEqual(response.status_code, 502)
        self.assertIn('credits and limits', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()