import os
from pathlib import Path

import requests
from flask import Blueprint, current_app, jsonify, request

translation_bp = Blueprint('translation', __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / 'prompt' / 'translate.txt'
POLYU_GENAI_URL = 'https://genai.comp.polyu.edu.hk/api/v1/chat/completions'
MODEL = 'DeepSeek-V4-Flash'
SUPPORTED_LANGUAGES = {
    'zh-CN': 'Simplified Chinese (简体中文)',
    'zh-TW': 'Traditional Chinese (繁體中文)',
}


@translation_bp.route('/translate', methods=['POST'])
def translate_note():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'A JSON request body is required.'}), 400

    text = data.get('text')
    target_language = data.get('target_language')
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Note text is required.'}), 400
    if len(text) > 20000:
        return jsonify({'error': 'Note text must be 20,000 characters or fewer.'}), 400
    if not isinstance(target_language, str) or target_language not in SUPPORTED_LANGUAGES:
        return jsonify({'error': 'Choose a supported target language.'}), 400

    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not api_key:
        return jsonify({'error': 'Translation is not configured. Set DEEPSEEK_API_KEY in .env.'}), 503

    try:
        system_prompt = PROMPT_PATH.read_text(encoding='utf-8').strip()
    except OSError:
        current_app.logger.error('Translation prompt file could not be read.')
        return jsonify({'error': 'Translation prompt is unavailable on the server.'}), 500
    if not system_prompt:
        return jsonify({'error': 'Translation prompt is empty.'}), 500

    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': (
                    f"Translate the following note into {SUPPORTED_LANGUAGES[target_language]}. "
                    f"Return only the translation.\n\n<note>\n{text}\n</note>"
                ),
            },
        ],
    }

    try:
        response = requests.post(
            POLYU_GENAI_URL,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=(5, 45),
        )
        response.raise_for_status()
        result = response.json()
        translated_text = result['choices'][0]['message']['content']
        if not isinstance(translated_text, str) or not translated_text.strip():
            raise ValueError('Empty translation response')
    except requests.Timeout:
        return jsonify({'error': 'The translation service timed out. Please try again.'}), 504
    except requests.HTTPError as error:
        status_code = getattr(error.response, 'status_code', 502)
        error_messages = {
            401: 'PolyU GenAI rejected the API key. Check DEEPSEEK_API_KEY in .env.',
            429: 'PolyU GenAI rate limit reached. Please wait and try again.',
        }
        message = error_messages.get(status_code, f'PolyU GenAI request failed (HTTP {status_code}).')
        current_app.logger.warning('Translation provider returned HTTP %s.', status_code)
        return jsonify({'error': message}), 502
    except requests.RequestException:
        current_app.logger.warning('Translation provider request failed.')
        return jsonify({'error': 'Could not connect to the translation service. Please try again.'}), 502
    except (ValueError, KeyError, IndexError, TypeError):
        current_app.logger.warning('Translation provider returned an invalid response.')
        return jsonify({'error': 'The translation service returned an invalid response.'}), 502

    return jsonify({'translated_text': translated_text.strip()})