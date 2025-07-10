
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

1. 将生成的代理文件 `external_whisper.py` 复制到 `open-webui/backend/open_webui/routers/` 目录下

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
               files = {"file": (os.path.basename(file_path), audio_file)}
               data = {}
               
               if metadata and metadata.get("language"):
                   data["language"] = metadata.get("language")
               
               r = requests.post(
                   f"{external_whisper_url}/transcribe",
                   files=files,
                   data=data
               )
               
               r.raise_for_status()
               response_data = r.json()
               
               # 保存转录结果
               transcript_file = f"{file_dir}/{id}.json"
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
                       detail = f"External Whisper: {res['error']}"
               except Exception:
                   detail = f"External Whisper: {e}"
           
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
