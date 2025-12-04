# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import pandas as pd
# from SwinFIR import SwinFIR
#

# image_path = "1.bmp"
# image = Image.open(image_path).convert('RGB')
# transform = transforms.Compose([transforms.ToTensor()])
# image = transform(image)  # 格式转换
# image = torch.reshape(image, (1, 3, 256, 256))
#

# model = SwinFIR()
# model.load_state_dict(torch.load("swin_fir.pth", map_location=torch.device('cpu')))
# model.eval()  # 模型转为测试阶段
#
# # 推断
# with torch.no_grad():
#     output = model(image)
#
# # 处理输出并保存为CSV文件
# output = output.squeeze().numpy()  # 如果输出维度为(1, 1, 256, 256)，去掉单一维度
# df = pd.DataFrame(output)
# df.to_csv('output_depth_map.csv', index=False, header=False)
# print(output)  # 输出结果

#使用最佳模型进行推断
import torch
from torch import nn
import pandas as pd
import cv2
from torchvision import transforms

from CAMEW_UNet import CAMEWUNet


def main():
    # 设置要使用的GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载最佳模型的权重
    best_model = CAMEWUNet()
    best_model = nn.DataParallel(best_model)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load("best_model.pth"))
    best_model.eval()

    # 示例代码，使用模型进行推断
    with torch.no_grad():
        # 读取输入图像
        img_path = "1.bmp"  # 图片名称为1.bmp，位于当前路径下
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道
        transform = transforms.Compose([
            transforms.ToTensor(),
            # 添加其他所需的变换
        ])
        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 添加 batch 维度

        output = best_model(img)
        # 处理输出并保存为CSV文件
        output = output.squeeze().cpu().numpy()  # 将张量移至 CPU，并转换为 NumPy 数组
        df = pd.DataFrame(output)
        df.to_csv('output_depth_map.csv', index=False, header=False)
        print(output)  # 输出结果

if __name__ == "__main__":
    main()




