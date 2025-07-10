
# �߾���Whisper����ʶ����񼯳�˵��

## ����1: ��װ����

```bash
pip install openai-whisper
```

## ����2: ����Whisper����

```bash
python whisper_service.py
```

## ����3: ���ɵ�Open WebUI

1. �����ɵĴ����ļ� `external_whisper.py` ���Ƶ� `open-webui/backend/open_webui/routers/` Ŀ¼��

2. �޸� `open-webui/backend/open_webui/main.py` �ļ�:

   a. ��ӵ������:
   ```python
   from open_webui.routers import external_whisper
   ```

   b. ���·��:
   ```python
   app.include_router(external_whisper.router, prefix="/api/v1/external_whisper", tags=["external_whisper"])
   ```

3. �޸� `open-webui/backend/open_webui/routers/audio.py` �ļ�:

   a. �� `transcription_handler` ��������Ӷ��ⲿWhisper�����֧��:
   ```python
   elif request.app.state.config.STT_ENGINE == "external_whisper":
       try:
           # ֱ��ʹ���ⲿWhisper�����URL
           external_whisper_url = request.app.state.config.EXTERNAL_WHISPER_URL
           if not external_whisper_url:
               raise Exception("�ⲿWhisper����URLδ����")
           
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
               
               # ����ת¼���
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

4. �޸� `open-webui/backend/open_webui/config.py` �ļ�������ⲿWhisper��������:
   ```python
   EXTERNAL_WHISPER_URL = PersistentConfig(
       "EXTERNAL_WHISPER_URL",
       "audio.stt.external_whisper_url",
       os.getenv("EXTERNAL_WHISPER_URL", ""),
   )
   ```

5. �޸� `open-webui/backend/open_webui/main.py` �ļ���������ü���:
   ```python
   app.state.config.EXTERNAL_WHISPER_URL = EXTERNAL_WHISPER_URL
   ```

6. ����Open WebUI����

7. �ڹ���������Ƶ�����У���STT��������Ϊ"external_whisper"���������ⲿWhisper����URL

## ע������

- ȷ��Whisper�����Open WebUI�����໥����
- ���ڴ�����Ƶ�ļ���������Ҫ������ʱ����
- �Ƽ�ʹ��large-v3ģ���Ի�����ʶ�𾫶�
