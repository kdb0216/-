import torch
from model import ViolenceClassifier
import torchvision.transforms as transforms
from PIL import Image
import os

class ViolenceClass:
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        初始化接口类，加载模型
        :param checkpoint_path: str, 训练好的模型权重文件的路径
        :param device: str, 指定设备，默认为'cuda:0'
        """
        self.device = device
        self.model = ViolenceClassifier.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, imgs: torch.Tensor) -> list:
        """
        对输入的图像进行分类
        :param imgs: torch.Tensor, 形状为`n*3*224*224`的tensor，输入图像已经归一化到0-1
        :return: list, 长度为`n`的python列表，每个值为对应的预测类别（0或1）
        """
        imgs = imgs.to(self.device)
        with torch.no_grad():
            logits = self.model(imgs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().tolist()
        return preds


if __name__ == "__main__":
    import os
    from PIL import Image

    # 加载并预处理图像
    img_dir = "D:\\pytorch\\violence_224\\test"
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    imgs = []
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for img_file in img_files:
        img = Image.open(img_file).convert("RGB")
        img_tensor = transformer(img)
        imgs.append(img_tensor)

    imgs_tensor = torch.stack(imgs)  # 形状为`n*3*224*224`的tensor

    # 实例化ViolenceClass并进行分类
    checkpoint_path = "train_logs/resnet18_pretrain_test/version_21/checkpoints/resnet18_pretrain_test-epoch=38-val_loss=0.06.ckpt"
    violence_classifier = ViolenceClass(checkpoint_path, device='cuda:0')
    predictions = violence_classifier.classify(imgs_tensor)

    # 打印预测结果
    i = 0
    for img_file, pred in zip(img_files, predictions):
        lable = int(img_file[29])
        print(f"Image: {img_file}, Predicted class: {pred}")
        if(lable==pred): i = i + 1

    print(i/146)