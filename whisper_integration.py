import os
import logging
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisper-integration")

# 加载环境变量
load_dotenv()

# 默认配置
DEFAULT_WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "http://localhost:8000")
DEFAULT_OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://localhost:8080")
DEFAULT_ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def test_whisper_service(whisper_service_url):
    """
    测试Whisper服务是否可用
    """
    try:
        response = requests.get(f"{whisper_service_url}/health")
        if response.status_code == 200 and response.json().get("status") == "ok":
            logger.info("Whisper服务可用")
            return True
        else:
            logger.error(f"Whisper服务健康检查失败: {response.text}")
            return False
    except Exception as e:
        logger.error(f"无法连接到Whisper服务: {e}")
        return False


def get_auth_token(openwebui_url, admin_email, admin_password):
    """
    获取Open WebUI的认证令牌
    """
    try:
        response = requests.post(
            f"{openwebui_url}/api/v1/auths/login",
            json={"email": admin_email, "password": admin_password}
        )
        
        if response.status_code == 200:
            token = response.json().get("token")
            if token:
                logger.info("成功获取认证令牌")
                return token
            else:
                logger.error("响应中没有令牌")
                return None
        else:
            logger.error(f"登录失败: {response.text}")
            return None
    except Exception as e:
        logger.error(f"登录过程中出错: {e}")
        return None


def update_audio_config(openwebui_url, token, whisper_service_url):
    """
    更新Open WebUI的音频配置，将STT引擎设置为外部Whisper服务
    """
    try:
        # 首先获取当前配置
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{openwebui_url}/api/v1/audio/config", headers=headers)
        
        if response.status_code != 200:
            logger.error(f"获取音频配置失败: {response.text}")
            return False
        
        current_config = response.json()
        
        # 修改配置，设置STT引擎为"external_whisper"
        # 这需要在Open WebUI中添加支持，我们将在后面的步骤中添加
        current_config["stt"]["ENGINE"] = "external_whisper"
        current_config["stt"]["EXTERNAL_WHISPER_URL"] = whisper_service_url
        
        # 更新配置
        response = requests.post(
            f"{openwebui_url}/api/v1/audio/config/update",
            json=current_config,
            headers=headers
        )
        
        if response.status_code == 200:
            logger.info("成功更新音频配置")
            return True
        else:
            logger.error(f"更新音频配置失败: {response.text}")
            return False
    except Exception as e:
        logger.error(f"更新音频配置过程中出错: {e}")
        return False


def create_proxy_endpoint(whisper_service_url):
    """
    创建一个代理端点，将请求转发到Whisper服务
    这个函数会生成一个Python文件，可以添加到Open WebUI的路由中
    """
    proxy_code = f'''
# 高精度Whisper语音识别代理
# 将此文件保存为 open-webui/backend/open_webui/routers/external_whisper.py

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

# Whisper服务URL
WHISPER_SERVICE_URL = "{whisper_service_url}"

@router.post("/transcriptions")
async def transcription(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    user=Depends(get_verified_user),
):
    """
    将音频转录请求代理到外部Whisper服务
    """
    try:
        # 保存上传的文件
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()
        filename = f"{{id}}.{{ext}}"
        
        file_dir = f"{{CACHE_DIR}}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{{file_dir}}/{{filename}}"
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # 准备发送到Whisper服务的请求
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            open(file_path, 'rb'),
            filename=filename,
            content_type=file.content_type
        )
        
        if language:
            form_data.add_field('language', language)
        
        # 发送请求到Whisper服务
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{{WHISPER_SERVICE_URL}}/transcribe",
                data=form_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log.error(f"Whisper服务返回错误: {{error_text}}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Whisper服务错误: {{error_text}}"
                    )
                
                result = await response.json()
                return {{
                    **result,
                    "filename": os.path.basename(file_path),
                }}
    
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理转录请求时出错: {{str(e)}}"
        )
'''

    # 保存代理端点代码到文件
    proxy_file_path = "external_whisper.py"
    with open(proxy_file_path, "w") as f:
        f.write(proxy_code)
    
    logger.info(f"代理端点代码已保存到 {proxy_file_path}")
    logger.info("请将此文件复制到 open-webui/backend/open_webui/routers/ 目录下")
    
    return proxy_file_path


def create_integration_instructions(proxy_file_path):
    """
    创建集成说明
    """
    instructions = f'''
# 高精度Whisper语音识别服务集成说明

## 步骤1: 安装依赖

```bash
pip install openai-whisper
```

## 步骤2: 启动Whisper服务

```bash
python whisper_service.py
```

## 步骤3: 集成到Open WebUI

1. 将生成的代理文件 `{proxy_file_path}` 复制到 `open-webui/backend/open_webui/routers/` 目录下

2. 修改 `open-webui/backend/open_webui/main.py` 文件:

   a. 添加导入语句:
   ```python
   from open_webui.routers import external_whisper
   ```

   b. 添加路由:
   ```python
   app.include_router(external_whisper.router, prefix="/api/v1/external_whisper", tags=["external_whisper"])
   ```

3. 修改 `open-webui/backend/open_webui/routers/audio.py` 文件:

   a. 在 `transcription_handler` 函数中添加对外部Whisper服务的支持:
   ```python
   elif request.app.state.config.STT_ENGINE == "external_whisper":
       try:
           # 直接使用外部Whisper服务的URL
           external_whisper_url = request.app.state.config.EXTERNAL_WHISPER_URL
           if not external_whisper_url:
               raise Exception("外部Whisper服务URL未配置")
           
           with open(file_path, "rb") as audio_file:
               files = {{"file": (os.path.basename(file_path), audio_file)}}
               data = {{}}
               
               if metadata and metadata.get("language"):
                   data["language"] = metadata.get("language")
               
               r = requests.post(
                   f"{{external_whisper_url}}/transcribe",
                   files=files,
                   data=data
               )
               
               r.raise_for_status()
               response_data = r.json()
               
               # 保存转录结果
               transcript_file = f"{{file_dir}}/{{id}}.json"
               with open(transcript_file, "w") as f:
                   json.dump(response_data, f)
               
               return response_data
       except Exception as e:
           log.exception(e)
           detail = None
           if r is not None:
               try:
                   res = r.json()
                   if "error" in res:
                       detail = f"External Whisper: {{res['error']}}"
               except Exception:
                   detail = f"External Whisper: {{e}}"
           
           raise Exception(detail if detail else "External Whisper: Server Connection Error")
   ```

4. 修改 `open-webui/backend/open_webui/config.py` 文件，添加外部Whisper服务配置:
   ```python
   EXTERNAL_WHISPER_URL = PersistentConfig(
       "EXTERNAL_WHISPER_URL",
       "audio.stt.external_whisper_url",
       os.getenv("EXTERNAL_WHISPER_URL", ""),
   )
   ```

5. 修改 `open-webui/backend/open_webui/main.py` 文件，添加配置加载:
   ```python
   app.state.config.EXTERNAL_WHISPER_URL = EXTERNAL_WHISPER_URL
   ```

6. 重启Open WebUI服务

7. 在管理界面的音频设置中，将STT引擎设置为"external_whisper"，并设置外部Whisper服务URL

## 注意事项

- 确保Whisper服务和Open WebUI可以相互访问
- 对于大型音频文件，可能需要调整超时设置
- 推荐使用large-v3模型以获得最佳识别精度
'''

    # 保存说明到文件
    instructions_file_path = "whisper_integration_instructions.md"
    with open(instructions_file_path, "w") as f:
        f.write(instructions)
    
    logger.info(f"集成说明已保存到 {instructions_file_path}")
    
    return instructions_file_path


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Open WebUI高精度Whisper语音识别集成工具")
    parser.add_argument("--whisper-url", default=DEFAULT_WHISPER_SERVICE_URL, help="Whisper服务URL")
    parser.add_argument("--openwebui-url", default=DEFAULT_OPENWEBUI_URL, help="Open WebUI URL")
    parser.add_argument("--admin-email", default=DEFAULT_ADMIN_EMAIL, help="管理员邮箱")
    parser.add_argument("--admin-password", default=DEFAULT_ADMIN_PASSWORD, help="管理员密码")
    parser.add_argument("--test-only", action="store_true", help="仅测试连接")
    parser.add_argument("--generate-only", action="store_true", help="仅生成集成文件")
    
    args = parser.parse_args()
    
    # 测试Whisper服务
    if not args.generate_only:
        whisper_service_available = test_whisper_service(args.whisper_url)
        if not whisper_service_available:
            logger.error("Whisper服务不可用，请确保服务已启动")
            if not args.test_only:
                logger.info("继续生成集成文件...")
        else:
            logger.info("Whisper服务测试成功")
    
    if args.test_only:
        return
    
    # 创建代理端点
    proxy_file_path = create_proxy_endpoint(args.whisper_url)
    
    # 创建集成说明
    instructions_file_path = create_integration_instructions(proxy_file_path)
    
    # 如果提供了管理员密码，尝试自动更新配置
    if args.admin_password and not args.generate_only:
        token = get_auth_token(args.openwebui_url, args.admin_email, args.admin_password)
        if token:
            update_success = update_audio_config(args.openwebui_url, token, args.whisper_url)
            if update_success:
                logger.info("已自动更新Open WebUI配置")
            else:
                logger.warning("无法自动更新Open WebUI配置，请按照说明手动更新")
        else:
            logger.warning("无法获取认证令牌，请按照说明手动更新配置")
    
    logger.info(f"集成文件已生成，请查看 {instructions_file_path} 获取详细说明")


if __name__ == "__main__":
    main() 