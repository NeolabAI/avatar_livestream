###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import enqueue_llm_response
from llm import llm_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager

import argparse
import random
import shutil
import asyncio
import os
import torch
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc


app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload
        

#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id) 
    opt_this.avatar_id = avatar_id
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')
    if (avatar_id and avatar_id != opt.avatar_id):
        # Avoid reloading if already cached globally
        if avatar_id not in global_avatars:
            global_avatars[avatar_id] = load_avatar(avatar_id)
        avatar_this = global_avatars[avatar_id]
    else:
        # Default avatar loaded at startup
        avatar_this = global_avatars.get(opt.avatar_id)
    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    avatar_session = registry.create("avatar", opt.model, opt=opt_this, model=model, avatar=avatar_this)
    return avatar_session

async def offer(request):
    try:
        return await rtc_manager.handle_offer(request)
    except Exception as exc:
        logger.exception("offer route exception:")
        return web.Response(
            content_type="application/json",
            status=500,
            text=json.dumps({"code": -1, "msg": f"offer failed: {exc}"}),
        )

async def on_shutdown(app):
    await rtc_manager.shutdown()


def _parse_gpu_ids(gpu_ids_raw: str):
    gpu_ids = []
    for chunk in gpu_ids_raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gid = int(chunk)
        if gid not in gpu_ids:
            gpu_ids.append(gid)
    return gpu_ids


def _parse_visible_gpu_map(visible_env: str):
    gpu_ids = []
    for chunk in visible_env.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gpu_ids.append(int(chunk))
    return gpu_ids


def configure_gpu_runtime(opt) -> None:
    """Configure GPU binding for either single-GPU or one-process multi-GPU mode."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; running on CPU.")
        opt.runtime_multi_gpu = False
        opt.runtime_gpu_ids = []
        opt.runtime_physical_gpu_ids = []
        opt.runtime_primary_gpu = None
        opt.runtime_physical_primary_gpu = None
        return

    if getattr(opt, "multi_gpu", False):
        gpu_ids = _parse_gpu_ids(opt.gpu_ids)
        if not gpu_ids:
            raise ValueError("--multi_gpu requires at least one id in --gpu_ids")
        visible_count = torch.cuda.device_count()
        invalid = [gid for gid in gpu_ids if gid < 0 or gid >= visible_count]
        if invalid:
            raise ValueError(f"Invalid gpu_ids={invalid}; visible GPU count is {visible_count}")

        primary_gpu = gpu_ids[0]
        torch.cuda.set_device(primary_gpu)
        opt.runtime_multi_gpu = len(gpu_ids) > 1
        opt.runtime_gpu_ids = gpu_ids
        opt.runtime_primary_gpu = primary_gpu
        opt.runtime_physical_gpu_ids = list(gpu_ids)
        opt.runtime_physical_primary_gpu = primary_gpu
        logger.info(
            "GPU runtime: multi_gpu=%s logical_gpu_ids=%s physical_gpu_ids=%s primary_logical=cuda:%s primary_physical_gpu=%s",
            opt.runtime_multi_gpu,
            gpu_ids,
            gpu_ids,
            primary_gpu,
            primary_gpu,
        )
        return

    # Single-GPU mode.
    visible_env = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env:
        visible_count = torch.cuda.device_count()
        if opt.gpu_id < 0 or opt.gpu_id >= visible_count:
            raise ValueError(
                f"gpu_id={opt.gpu_id} out of visible range [0, {visible_count - 1}] "
                f"for CUDA_VISIBLE_DEVICES={visible_env}"
            )
        visible_gpu_map = _parse_visible_gpu_map(visible_env)
        physical_gpu = (
            visible_gpu_map[opt.gpu_id]
            if opt.gpu_id < len(visible_gpu_map)
            else opt.gpu_id
        )
        torch.cuda.set_device(opt.gpu_id)
        opt.runtime_multi_gpu = False
        opt.runtime_gpu_ids = [opt.gpu_id]
        opt.runtime_primary_gpu = opt.gpu_id
        opt.runtime_physical_gpu_ids = [physical_gpu]
        opt.runtime_physical_primary_gpu = physical_gpu
        logger.info(
            "GPU runtime: single_gpu logical=cuda:%s physical_gpu=%s via CUDA_VISIBLE_DEVICES=%s (%s)",
            opt.gpu_id,
            physical_gpu,
            visible_env,
            torch.cuda.get_device_name(opt.gpu_id),
        )
        return

    # No visibility mask: bind directly to the requested physical GPU.
    visible_count = torch.cuda.device_count()
    if opt.gpu_id < 0 or opt.gpu_id >= visible_count:
        raise ValueError(
            f"gpu_id={opt.gpu_id} out of visible range [0, {visible_count - 1}]"
        )
    torch.cuda.set_device(opt.gpu_id)
    opt.runtime_multi_gpu = False
    opt.runtime_gpu_ids = [opt.gpu_id]
    opt.runtime_primary_gpu = opt.gpu_id
    opt.runtime_physical_gpu_ids = [opt.gpu_id]
    opt.runtime_physical_primary_gpu = opt.gpu_id
    logger.info(
        "GPU runtime: single_gpu logical=cuda:%s physical_gpu=%s direct_bind (%s)",
        opt.gpu_id,
        opt.gpu_id,
        torch.cuda.get_device_name(opt.gpu_id),
    )


def maybe_enable_model_parallel(opt, model):
    if not getattr(opt, "runtime_multi_gpu", False):
        return model
    device_ids = getattr(opt, "runtime_gpu_ids", [])
    if len(device_ids) < 2:
        return model

    primary = device_ids[0]
    if opt.model == "musetalk":
        vae, unet, pe, timesteps, audio_processor = model
        root_gpu = device_ids[0]
        root_device = torch.device(f"cuda:{root_gpu}")
        output_gpu = device_ids[-1]
        output_device = torch.device(f"cuda:{output_gpu}")
        mode = getattr(opt, "musetalk_multi_gpu_mode", "data_parallel")

        # Mode 1: UNet DataParallel (legacy/default).
        if mode == "data_parallel":
            # DataParallel requires the source module on device_ids[0].
            unet.device = root_device
            unet.model = unet.model.to(root_device)
            unet.model = torch.nn.DataParallel(unet.model, device_ids=device_ids, output_device=output_gpu)
            pe = pe.to(root_device)
            timesteps = timesteps.to(root_device)
            # Move VAE decode to the output GPU to reduce pressure on GPU0.
            vae.vae = vae.vae.to(output_device)
            vae.device = output_device
            logger.info(
                "Enabled MuseTalk DataParallel on GPUs %s (unet_output=cuda:%s, vae_decode=cuda:%s)",
                device_ids,
                output_gpu,
                output_gpu,
            )
            return vae, unet, pe, timesteps, audio_processor

        # Mode 2: Split workers by stage (no UNet scatter/gather).
        # This avoids DataParallel overhead on small realtime batches.
        if mode == "split_workers":
            unet.device = root_device
            unet.model = unet.model.to(root_device)
            pe = pe.to(root_device)
            timesteps = timesteps.to(root_device)
            vae.vae = vae.vae.to(output_device)
            vae.device = output_device
            logger.info(
                "Enabled MuseTalk split_workers on GPUs %s (unet=cuda:%s, vae_decode=cuda:%s)",
                device_ids,
                root_gpu,
                output_gpu,
            )
            return vae, unet, pe, timesteps, audio_processor

        logger.warning(
            "Unknown musetalk_multi_gpu_mode=%s; fallback to data_parallel.",
            mode,
        )
        unet.device = root_device
        unet.model = unet.model.to(root_device)
        unet.model = torch.nn.DataParallel(unet.model, device_ids=device_ids, output_device=output_gpu)
        pe = pe.to(root_device)
        timesteps = timesteps.to(root_device)
        vae.vae = vae.vae.to(output_device)
        vae.device = output_device
        return vae, unet, pe, timesteps, audio_processor

    if opt.model == "wav2lip":
        primary_device = torch.device(f"cuda:{primary}")
        model = model.to(primary_device)
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=primary)
        logger.info("Enabled Wav2Lip DataParallel on GPUs %s", device_ids)
        return model

    logger.warning("multi_gpu is not implemented for model=%s; fallback to single GPU.", opt.model)
    return model



def main():
    global rtc_manager, opt, model,load_avatar
    # 解析命令行参数
    from config import parse_args
    opt = parse_args()
    configure_gpu_runtime(opt)
    if (
        opt.model == "musetalk"
        and getattr(opt, "runtime_multi_gpu", False)
        and getattr(opt, "musetalk_multi_gpu_mode", "data_parallel") == "data_parallel"
        and opt.batch_size > 8
    ):
        logger.warning(
            "MuseTalk data_parallel with batch_size=%s may increase realtime latency. "
            "Try --batch_size 4 or --musetalk_multi_gpu_mode split_workers.",
            opt.batch_size,
        )

    # ─── 加载 avatar 插件（触发 @register 注册）──────────────────────
    _avatar_modules = {
        'musetalk':   'avatars.musetalk_avatar',
        'wav2lip':    'avatars.wav2lip_avatar',
        'ultralight': 'avatars.ultralight_avatar',
    }
    import importlib
    avatar_mod = importlib.import_module(_avatar_modules[opt.model])
    load_model = avatar_mod.load_model
    load_avatar = avatar_mod.load_avatar
    warm_up = avatar_mod.warm_up
    logger.info(opt)

    if opt.model == 'musetalk':
        model = load_model()
        model = maybe_enable_model_parallel(opt, model)
        try:
            global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        except Exception as _e:
            logger.warning("Startup avatar '%s' load failed (%s). UI must pass an 'avatar' param at session creation.", opt.avatar_id, _e)
            global_avatars[opt.avatar_id] = None
        warm_up(opt.batch_size, model)
    elif opt.model == 'wav2lip':
        model = load_model("./models/wav2lip.pth")
        model = maybe_enable_model_parallel(opt, model)
        try:
            global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        except Exception as _e:
            logger.warning("Startup avatar '%s' load failed (%s). UI must pass an 'avatar' param at session creation.", opt.avatar_id, _e)
            global_avatars[opt.avatar_id] = None
        warm_up(opt.batch_size, model, 256)
    elif opt.model == 'ultralight':
        model = load_model(opt)
        global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,global_avatars[opt.avatar_id],160)

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
    # share avatar_sessions (RTCManager handles it but routes.py expects it)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["opt"] = opt
    appasync["llm_response"] = llm_response
    appasync["llm_enqueue_response"] = enqueue_llm_response

    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    
    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync) 

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        # Reap PeerConnections whose peer vanished without a clean ICE close
        # (leaks slots until LIVETALKING_MAX_SESSIONS -> "reach max session").
        rtc_manager.start_reaper()
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
    
    
    
