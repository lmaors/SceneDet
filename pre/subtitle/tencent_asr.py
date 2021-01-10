from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models
import base64
import io
import sys
import time
from add_punctuator import reduce_punctuator
if sys.version_info[0] == 3:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def audio2text(audio_path):
    # 本地文件方式请求
    try:

        cred = credential.Credential("AKIDgBUcd5sK6d08Ac70rUs2zrgGcieal2Rb", "82hR3TeaHm22wlV8fw8Edv695vcCfzNm")
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        clientProfile.signMethod = "TC3-HMAC-SHA256"
        client = asr_client.AsrClient(cred, "ap-shanghai", clientProfile)

        with open(audio_path, "rb") as f:
            if sys.version_info[0] == 2:
                content = base64.b64encode(f.read())
            else:
                content = base64.b64encode(f.read()).decode('utf-8')

        req = models.CreateRecTaskRequest()
        params = {"ChannelNum": 1, "ResTextFormat": 0, "SourceType": 1}
        req._deserialize(params)
        req.EngineModelType = "16k_zh"
        req.Data = content

        resp = client.CreateRecTask(req)
        print('使用腾讯语音识别api转换')
        # print(resp.to_json_string())

        for i in range(10):
            print('语音识别结果查询中...')

            req_1 = models.DescribeTaskStatusRequest()
            task_id = str(resp.Data)
            req_1.from_json_string(task_id)
            resp = client.DescribeTaskStatus(req_1)
            # print(resp.to_json_string())
            if resp.Data.StatusStr == 'success':
                result = resp.Data.Result
                if result:
                    t = result.split(']')[0].split('[')[-1]
                    txt = result.split(']')[-1].strip()
                    print('result: {}'.format(txt))
                    punct_txt = reduce_punctuator(txt)
                    print('punct_result: {}'.format(punct_txt))
                    return punct_txt

                else:
                    print('未检测到语音文本')
                break
            sys.stdout.flush()
            time.sleep(2)

    except TencentCloudSDKException as err:
        print(err)

if __name__ == '__main__':
    path = '/home/lcc/Labworking/SceneSeg/data/myexp/aud_wav/demo/shot_0003.wav'
    text = audio2text(path)
    print(text)