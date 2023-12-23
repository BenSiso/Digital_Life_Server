import argparse

import torch


def str2bool(v):
    if v is None:
        # 当传入None时，可以选择返回False或者抛出异常
        return False  # 或者 raise argparse.ArgumentTypeError("None value is not supported.")

    # 去除字符串两端的空格
    v = v.strip()

    # 处理常见的True值
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    # 处理常见的False值
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    # 处理空字符串或者其他不支持的值
    elif not v:
        # 选择返回False或者抛出异常
        return False  # 或者 raise argparse.ArgumentTypeError("Empty string is not a valid boolean value.")
    else:
        raise argparse.ArgumentTypeError(f"Unsupported value encountered: '{v}'（遇到不支持的值：'{v}'）")


def get_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--APIKey", type=str, nargs='?', required=False)
    # ERNIEBot app SecretKey
    parser.add_argument("--SecretKey", type=str, nargs='?', required=False)
    # ERNIEBot accessToken
    parser.add_argument("--accessToken", type=str, nargs='?', required=False)
    # ChatGPT 代理服务器 http://127.0.0.1:7890
    parser.add_argument("--proxy", type=str, nargs='?', required=False)
    # 会话模型
    parser.add_argument("--model", type=str, nargs='?', required=True, default="paimon")
    # 流式语音
    parser.add_argument("--stream", type=str2bool, nargs='?', required=False, default=True)
    # 角色 ： paimon、 yunfei、 catmaid
    parser.add_argument("--character", type=str, nargs='?', required=True)
    # parser.add_argument("--ip", type=str, nargs='?', required=False)
    # 洗脑模式。循环发送提示词
    parser.add_argument("--brainwash", type=str2bool, nargs='?', required=False, default=False)
    # 定义运行的端口号
    parser.add_argument("--port", type=str, nargs='?', required=False, default=38438)
    # 是否使用whisper
    parser.add_argument("--faster_whisper", type=bool, action='store_true', required=False)
    parser.add_argument("--whisper", type=bool, action='store_true', required=False)
    parser.add_argument("--flash_attn", type=bool, action='store_true', required=False)
    parser.add_argument("--whisper_model", type=str, required=False)

    args = parser.parse_args()

    if args.whisper or args.faster_whisper:
        assert args.whisper_model is not None, "whisper_model is required when using whisper (or faster whisper)"
        if args.faster_whisper:
            args.whisper = False
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {args.device}')

    return args
