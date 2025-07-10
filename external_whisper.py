
# �߾���Whisper����ʶ�����
# �����ļ�����Ϊ open-webui/backend/open_webui/routers/external_whisper.py

import logging
import os
import uuid
import aiohttp
import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from open_webui.utils.auth import get_verified_user
from open_webui.config import CACHE_DIR

router = APIRouter()

log = logging.getLogger(__name__)

# Whisper����URL
WHISPER_SERVICE_URL = "http://localhost:8000"

@router.post("/transcriptions")
async def transcription(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    user=Depends(get_verified_user),
):
    """
    ����Ƶת¼��������ⲿWhisper����
    """
    try:
        # �����ϴ����ļ�
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()
        filename = f"{id}.{ext}"
        
        file_dir = f"{CACHE_DIR}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # ׼�����͵�Whisper���������
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            open(file_path, 'rb'),
            filename=filename,
            content_type=file.content_type
        )
        
        if language:
            form_data.add_field('language', language)
        
        # ��������Whisper����
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WHISPER_SERVICE_URL}/transcribe",
                data=form_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"Whisper���񷵻ش���: {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Whisper�������: {error_text}"
                    )
                
                result = await response.json()
                return {
                    **result,
                    "filename": os.path.basename(file_path),
                }
    
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"����ת¼����ʱ����: {str(e)}"
        )
