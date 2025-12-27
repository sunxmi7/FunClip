#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
FunClip API Server
RESTful API wrapper for FunClip video clipping functionality
"""

import os
import sys
import json
import logging
import argparse
import traceback
import uuid
import requests
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add funclip directory for utils imports
funclip_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'funclip')
if funclip_dir not in sys.path:
    sys.path.insert(0, funclip_dir)

from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from funasr import AutoModel
from funclip.videoclipper import VideoClipper
import numpy as np

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def clean_state_for_json(state):
    """Remove non-serializable objects from state"""
    if not state:
        return {}
    cleaned = {}
    for k, v in state.items():
        if k in ['video']:  # 移除VideoFileClip等无法序列化的对象
            continue
        if isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, dict):
            cleaned[k] = clean_state_for_json(v)
        else:
            cleaned[k] = v
    return cleaned

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Global variables for model
audio_clipper = None
MODEL_LOADED = False

# Default prompts
DEFAULT_SYSTEM_PROMPT = "你是一个视频srt字幕分析剪辑器，输入视频的srt字幕，分析其中的精彩且尽可能连续的片段并裁剪出来，输出四条以内的片段，将片段中在时间上连续的多个句子及它们的时间戳合并为一条，注意确保文字与时间戳的正确匹配。输出需严格按照如下格式：1. [开始时间-结束时间] 文本，注意其中的连接符是\"-\""
DEFAULT_USER_PROMPT = "这是待裁剪的视频srt字幕："

# LLM API imports
from funclip.llm.openai_api import openai_call
from funclip.llm.qwen_api import call_qwen_model
from funclip.llm.g4f_openai_api import g4f_openai_call
from funclip.utils.trans_utils import extract_timestamps


def init_model(lang='zh'):
    """Initialize FunASR model"""
    global audio_clipper, MODEL_LOADED

    if MODEL_LOADED:
        return audio_clipper

    try:
        logger.info(f"Initializing FunASR model for language: {lang}")
        if lang == 'zh':
            funasr_model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            )
        else:
            funasr_model = AutoModel(
                model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            )

        audio_clipper = VideoClipper(funasr_model)
        audio_clipper.lang = lang
        MODEL_LOADED = True
        logger.info("Model initialized successfully")
        return audio_clipper
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise


def llm_inference(system_content, user_content, srt_text, model, apikey):
    """Call LLM for intelligent clipping analysis"""
    SUPPORT_LLM_PREFIX = ['qwen', 'gpt', 'g4f', 'moonshot', 'deepseek']

    if model.startswith('qwen'):
        return call_qwen_model(apikey, model, user_content + '\n' + srt_text, system_content)
    if model.startswith('gpt') or model.startswith('moonshot') or model.startswith('deepseek'):
        return openai_call(apikey, model, system_content, user_content + '\n' + srt_text)
    elif model.startswith('g4f'):
        model_name = "-".join(model.split('-')[1:])
        return g4f_openai_call(model_name, system_content, user_content + '\n' + srt_text)
    else:
        raise ValueError(f"Unsupported LLM model: {model}. Supported prefixes: {SUPPORT_LLM_PREFIX}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/v1/recognize', methods=['POST'])
def recognize():
    """ASR recognition endpoint"""
    try:
        # Handle both JSON and multipart/form-data
        if request.is_json:
            data = request.json
            video_path = data.get('video_path')
            hotwords = data.get('hotwords', '')
            output_dir = data.get('output_dir', '')
            distinguish_speaker = data.get('distinguish_speaker', False)
        else:
            video_path = None
            hotwords = request.form.get('hotwords', '')
            output_dir = request.form.get('output_dir', '')
            distinguish_speaker = request.form.get('distinguish_speaker', 'false').lower() == 'true'

            # Get uploaded file
            if 'file' in request.files:
                file = request.files['file']
                if file.filename:
                    # Use environment variable or default to match Go config
                    upload_dir = os.environ.get('UPLOAD_DIR', '/home/sunxiaomeng/Large_Model_Registration/backend/uploads')
                    video_path = os.path.join(upload_dir, f'funclip_{uuid.uuid4().hex}_{file.filename}')
                    os.makedirs(os.path.dirname(video_path), exist_ok=True)
                    file.save(video_path)

        if not video_path or not os.path.exists(video_path):
            return jsonify({
                'code': -1,
                'message': 'Video file not found',
                'data': None
            }), 400

        logger.info(f"Recognizing video: {video_path}")

        # Initialize model if not already loaded
        clipper = init_model()

        # Perform recognition
        sd_switch = 'Yes' if distinguish_speaker else 'No'
        res_text, res_srt, state = clipper.video_recog(
            video_path,
            sd_switch=sd_switch,
            hotwords=hotwords,
            output_dir=output_dir if output_dir else None
        )

        return jsonify({
            'code': 0,
            'message': 'Recognition successful',
            'data': {
                'text': res_text,
                'srt': res_srt,
                'state': clean_state_for_json(state)  # 清理后返回
            }
        })

    except Exception as e:
        logger.error(f"Recognition error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'code': -1,
            'message': f'Recognition failed: {str(e)}',
            'data': None
        }), 500


@app.route('/api/v1/clip', methods=['POST'])
def clip():
    """Video clipping endpoint based on text/speaker"""
    try:
        data = request.json
        video_path = data.get('video_path')
        dest_text = data.get('dest_text', '')
        dest_spk = data.get('dest_spk', '')
        start_ost = data.get('start_ost', 0)
        end_ost = data.get('end_ost', 100)
        output_dir = data.get('output_dir', '')
        state = data.get('state')
        timestamps = data.get('timestamps', [])
        add_subtitle = data.get('add_subtitle', False)
        font_size = data.get('font_size', 32)
        font_color = data.get('font_color', 'white')

        if not video_path or not os.path.exists(video_path):
            return jsonify({
                'code': -1,
                'message': 'Video file not found',
                'data': None
            }), 400

        if not state:
            return jsonify({
                'code': -1,
                'message': 'Missing recognition state. Call /api/v1/recognize first.',
                'data': None
            }), 400

        logger.info(f"Clipping video: {video_path}")

        # Initialize model if not already loaded
        clipper = init_model()

        # Load video for state
        import moviepy.editor as mpy
        state['video_filename'] = video_path
        state['video'] = mpy.VideoFileClip(video_path)

        # 转换时间戳格式：从毫秒到帧索引
        timestamp_list = None
        if timestamps and len(timestamps) > 0:
            timestamp_list = []
            for ts in timestamps:
                start_ms = float(ts.get('start', 0))
                end_ms = float(ts.get('end', 0))
                timestamp_list.append([start_ms, end_ms])

        # Perform clipping
        if add_subtitle:
            clip_video_file, message, clip_srt = clipper.video_clip(
                dest_text, start_ost, end_ost, state,
                dest_spk=dest_spk,
                output_dir=output_dir if output_dir else None,
                font_size=font_size,
                font_color=font_color,
                add_sub=True,
                timestamp_list=timestamp_list
            )
        else:
            clip_video_file, message, clip_srt = clipper.video_clip(
                dest_text, start_ost, end_ost, state,
                dest_spk=dest_spk,
                output_dir=output_dir if output_dir else None,
                add_sub=False,
                timestamp_list=timestamp_list
            )

        # 保存 SRT 文件到 output_dir
        subtitle_path = None
        if clip_srt and clip_video_file and output_dir:
            subtitle_path = clip_video_file.replace('.mp4', '.srt')
            os.makedirs(output_dir, exist_ok=True)
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write(clip_srt)
            logger.info(f"Saved subtitle to: {subtitle_path}")

        return jsonify({
            'code': 0,
            'message': 'Clipping successful',
            'data': {
                'output_path': clip_video_file,
                'subtitle_path': subtitle_path if subtitle_path else (clip_video_file.replace('.mp4', '.srt') if clip_video_file else None),
                'log': message,
                'srt_content': clip_srt
            }
        })

    except Exception as e:
        logger.error(f"Clipping error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'code': -1,
            'message': f'Clipping failed: {str(e)}',
            'data': None
        }), 500


@app.route('/api/v1/llm-inference', methods=['POST'])
def llm_inference_api():
    """LLM inference endpoint for intelligent clipping"""
    try:
        data = request.json
        srt_text = data.get('srt_text', '')
        llm_model = data.get('llm_model', 'qwen-plus')
        llm_api_key = data.get('llm_api_key', '')
        system_prompt = data.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
        user_prompt = data.get('user_prompt', DEFAULT_USER_PROMPT)

        if not srt_text:
            return jsonify({
                'code': -1,
                'message': 'SRT text is required',
                'data': None
            }), 400

        if not llm_api_key:
            return jsonify({
                'code': -1,
                'message': 'LLM API key is required',
                'data': None
            }), 400

        logger.info(f"Running LLM inference with model: {llm_model}")

        # Call LLM
        llm_result = llm_inference(
            system_prompt,
            user_prompt,
            srt_text,
            llm_model,
            llm_api_key
        )

        return jsonify({
            'code': 0,
            'message': 'LLM inference successful',
            'data': {
                'llm_result': llm_result,
                'timestamps': extract_timestamps(llm_result)
            }
        })

    except Exception as e:
        logger.error(f"LLM inference error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'code': -1,
            'message': f'LLM inference failed: {str(e)}',
            'data': None
        }), 500


@app.route('/api/v1/gradio-inference', methods=['POST'])
def gradio_inference_api():
    """Call Gradio /run/predict for LLM analysis"""
    try:
        data = request.json
        srt_content = data.get('srt_content', '')
        llm_model = data.get('llm_model', 'qwen-plus')
        llm_api_key = data.get('llm_api_key', '')
        system_prompt = data.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
        gradio_url = data.get('gradio_url', 'http://127.0.0.1:7860')

        if not srt_content:
            return jsonify({
                'code': -1,
                'message': 'SRT content is required',
                'data': None
            }), 400

        # Prepare Gradio request data
        gradio_data = {
            "data": [
                system_prompt,
                srt_content,
                srt_content,
                llm_model,
                llm_api_key
            ],
            "fn_index": 7,
            "session_hash": str(uuid.uuid4().replace('-', '')[:8])
        }

        logger.info(f"Calling Gradio inference at {gradio_url}/run/predict")

        # Call Gradio API
        response = requests.post(
            f"{gradio_url}/run/predict",
            json=gradio_data,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                'code': -1,
                'message': f'Gradio API error: {response.text}',
                'data': None
            }), 500

        result = response.json()
        llm_result = result.get('data', [''])[0] if result.get('data') else ''

        return jsonify({
            'code': 0,
            'message': 'Gradio inference successful',
            'data': {
                'llm_result': llm_result,
                'timestamps': extract_timestamps(llm_result)
            }
        })

    except Exception as e:
        logger.error(f"Gradio inference error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'code': -1,
            'message': f'Gradio inference failed: {str(e)}',
            'data': None
        }), 500


@app.route('/api/v1/auto-clip', methods=['POST'])
def auto_clip():
    """Automatic video clipping endpoint (ASR + LLM + Clip)"""
    try:
        data = request.json
        video_path = data.get('video_path')
        hotwords = data.get('hotwords', '')
        output_dir = data.get('output_dir', '')
        llm_model = data.get('llm_model', 'qwen-plus')
        llm_api_key = data.get('llm_api_key', '')
        system_prompt = data.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
        user_prompt = data.get('user_prompt', DEFAULT_USER_PROMPT)

        if not video_path or not os.path.exists(video_path):
            return jsonify({
                'code': -1,
                'message': 'Video file not found',
                'data': None
            }), 400

        if not llm_api_key:
            return jsonify({
                'code': -1,
                'message': 'LLM API key is required',
                'data': None
            }), 400

        logger.info(f"Starting auto-clip for video: {video_path}")
        logs = []

        # Step 1: ASR Recognition
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Starting ASR recognition...")
        clipper = init_model()
        res_text, res_srt, state = clipper.video_recog(
            video_path,
            sd_switch='No',
            hotwords=hotwords,
            output_dir=output_dir if output_dir else None
        )
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: ASR recognition completed")
        logs.append(f"Recognized text: {res_text[:100]}...")

        # Step 2: LLM Analysis
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Running LLM analysis...")
        llm_result = llm_inference(
            system_prompt,
            user_prompt,
            res_srt,
            llm_model,
            llm_api_key
        )
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: LLM analysis completed")
        logs.append(f"LLM result: {llm_result}")

        # Step 3: Extract timestamps and clip
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Extracting timestamps and clipping...")
        timestamp_list = extract_timestamps(llm_result)
        logs.append(f"Extracted {len(timestamp_list)} timestamp segments")

        # Load video for clipping
        import moviepy.editor as mpy
        state['video_filename'] = video_path
        state['video'] = mpy.VideoFileClip(video_path)

        # Perform AI clipping
        clip_video_file, message, clip_srt = clipper.video_clip(
            '', 0, 0, state,
            output_dir=output_dir if output_dir else None,
            timestamp_list=timestamp_list,
            add_sub=False
        )

        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Clipping completed")
        logs.append(f"Clipping log: {message}")
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Auto-clip process completed successfully!")

        return jsonify({
            'code': 0,
            'message': 'Auto-clip completed successfully',
            'data': {
                'output_path': clip_video_file,
                'subtitle_path': clip_video_file.replace('.mp4', '.srt') if clip_video_file else None,
                'log': '\n'.join(logs),
                'srt_content': clip_srt,
                'recognized_text': res_text,
                'llm_result': llm_result
            }
        })

    except Exception as e:
        logger.error(f"Auto-clip error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'code': -1,
            'message': f'Auto-clip failed: {str(e)}',
            'data': {
                'log': traceback.format_exc()
            }
        }), 500


def main():
    parser = argparse.ArgumentParser(description='FunClip API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'], help='Language model')
    parser.add_argument('--preload', action='store_true', help='Preload model on startup')
    args = parser.parse_args()

    # Preload model if requested
    if args.preload:
        logger.info("Preloading model...")
        init_model(args.lang)

    logger.info(f"Starting FunClip API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
