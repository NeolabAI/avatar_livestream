var pc = null;

function getSelectedAvatarId() {
    var avatarSelect = document.getElementById('avatar-select');
    if (!avatarSelect) {
        return '';
    }
    return String(avatarSelect.value || '').trim();
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postJsonWithRetry(url, payload, options = {}) {
    const retries = Number.isInteger(options.retries) ? options.retries : 2;
    const delayMs = Number.isInteger(options.delayMs) ? options.delayMs : 700;
    const timeoutMs = Number.isInteger(options.timeoutMs) ? options.timeoutMs : 12000;

    let lastError = null;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, {
                body: JSON.stringify(payload),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST',
                signal: controller.signal,
            });
            clearTimeout(timer);
            return response;
        } catch (err) {
            clearTimeout(timer);
            lastError = err;
            if (attempt < retries) {
                await sleep(delayMs * (attempt + 1));
            }
        }
    }
    throw lastError || new Error('Unknown network error');
}

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete (max 3s timeout to avoid hanging)
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                var timer = setTimeout(resolve, 3000);
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        clearTimeout(timer);
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(async () => {
        var offer = pc.localDescription;
        const avatarId = getSelectedAvatarId();
        // Diagnostic: confirm which avatar is bound to the new session.
        // build_avatar_session reads params['avatar']; empty -> server default.
        console.log('[LiveTalking] /offer avatar =', JSON.stringify(avatarId) || '(empty -> default)');
        // /offer builds the avatar session server-side (load face_imgs + coords +
        // precompute silence frames). A 361-frame avatar builds in ~9s, but a
        // 1520-frame avatar takes ~35-41s — longer than the old 15s timeout, so the
        // AbortController fired ("signal is aborted without reason") and the retry
        // then created ANOTHER session. So: give one generous timeout (large avatars
        // can build ~40s; build_lock serializes concurrent offers so a 2nd connect
        // queues behind the 1st) and do NOT retry (a retry never rescues a slow
        // build, it just leaks more sessions + PCs).
        const response = await postJsonWithRetry('/offer', {
            avatar: avatarId,
            sdp: offer.sdp,
            type: offer.type,
        }, {
            retries: 0,
            delayMs: 800,
            timeoutMs: 90000,
        });
        if (!response.ok) {
            const bodyText = await response.text().catch(() => '');
            throw new Error('/offer failed: HTTP ' + response.status + ' ' + bodyText);
        }
        return response;
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        // Server rejects offers over the session cap with {code:-1,msg:"reach max session"}
        // (HTTP 200). Surface that clearly instead of letting setRemoteDescription fail
        // with a confusing "undefined sdp" error.
        if (answer && answer.code && answer.code !== 0) {
            const why = answer.msg || ('code ' + answer.code);
            throw new Error(why);
        }
        document.getElementById('sessionid').value = answer.sessionid
        if (typeof window.onLiveTalkingSessionReady === 'function') {
            window.onLiveTalkingSessionReady(answer.sessionid);
        }
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        const message = (e && e.message) ? e.message : String(e);
        const friendly = /reach max session/i.test(message)
            ? 'Đã có quá nhiều kết nối đang mở (server cap). Bấm "Dừng" rồi kết nối lại, hoặc khởi động lại server.'
            : ('Start session failed: ' + message);
        // Roll back the local pc so a retry creates a fresh one.
        try { if (pc && pc.connectionState !== 'closed') pc.close(); } catch (_) {}
        var startBtn = document.getElementById('start');
        var stopBtn = document.getElementById('stop');
        if (startBtn) startBtn.style.display = 'inline-block';
        if (stopBtn) stopBtn.style.display = 'none';
        alert(friendly);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    var useStunEl = document.getElementById('use-stun');
    if (useStunEl && useStunEl.checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    // Close any existing PeerConnection before creating a new one. Without this,
    // selecting a different avatar and pressing Connect again (without Stop)
    // orphans the old pc — the server-side PC stays "connected" and leaks a
    // session slot until the LIVETALKING_MAX_SESSIONS cap is hit.
    if (pc && pc.connectionState !== 'closed') {
        var oldSid = document.getElementById('sessionid').value || '';
        if (typeof window.onLiveTalkingSessionStopped === 'function' && oldSid) {
            window.onLiveTalkingSessionStopped(oldSid);
        }
        try { pc.close(); } catch (_) {}
        // Give the server a moment to see the ICE teardown before we open a new pc.
    }

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            var v = document.getElementById('video');
            if (v) v.srcObject = evt.streams[0];
        } else {
            var a = document.getElementById('audio');
            if (a) a.srcObject = evt.streams[0];
        }
    });

    var startBtn = document.getElementById('start');
    var stopBtn = document.getElementById('stop');
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-block';
    return negotiate();
}

function stop() {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    var stopBtn = document.getElementById('stop');
    if (stopBtn) stopBtn.style.display = 'none';

    // close peer connection
    setTimeout(() => {
        pc.close();
    }, 500);
}

window.onunload = function(event) {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
    var sid = document.getElementById('sessionid').value || '';
    if (typeof window.onLiveTalkingSessionStopped === 'function' && sid) {
        window.onLiveTalkingSessionStopped(sid);
    }
    setTimeout(() => {
        pc.close();
    }, 500);
    e = e || window.event;
    if (e) {
        e.returnValue = 'Đóng trang';
    }
    return 'Đóng trang';
}
